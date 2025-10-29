from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import cv2
import logging
import os
import math
import random
import uvicorn
import numpy as np

app = FastAPI(title="Aspect Ratio Finder and Same Video Detector")
log = logging.getLogger("uvicorn.error")

# In-memory storage of videos where the key is the video ID and the value is the video metadata
# The video metadata is a dictionary with the following keys:
# - video_id: the ID of the video
# - width: the width of the video
# - height: the height of the video
# - aspect_ratio: the aspect ratio of the video
# - ratio_bucket: the ratio bucket of the video
# - filename: the filename of the video
# - frame_hashes: list of perceptual frame hashes for similarity detection
videos = {}

# list of perceptual frame hashes for similarity detection with key:value
# video_id: list of hashes
frame_hash_dict = {}

# Aspect ratio buckets to put videos into with 1% tolerance
ASPECT_RATIO_BUCKETS = {
    "9:16": 9 / 16,
    "1:1": 1.0,
    "4:5": 4 / 5,
    "16:9": 16 / 9,
    "Other": 0
}

def get_next_video_id() -> str:
    """
    Generate a random 8-digit video ID
    :return: 8-digit number if a random ID could be generated. Raises exception otherwise.
    :rtype: str
    """
    max_id = 99999999
    for i in range(max_id+1):
        random_id = random.randint(0, max_id)
        video_id = f"{random_id:08d}"
        if video_id not in videos.keys():
            return video_id
    raise Exception("No free video IDs available")

def find_ratio_bucket(width, height):
    """
    Find the ratio bucket for the given aspect ratio.
    :param int width: width of the video we are dealing with
    :param int height: height of the video we are dealing with
    :return: The ratio bucket name, or Other if no bucket is within 1% tolerance.
    :rtype: str
    """
    aspect_ratio_float = width / height if height != 0 else 0
    for name, ratio in ASPECT_RATIO_BUCKETS.items():
        tolerance = ratio * 0.01  # 1% tolerance
        if abs(aspect_ratio_float - ratio) <= tolerance:
            return name
    return "Other"

def dhash_gray(image_gray: np.ndarray) -> int:
    """
    Compute a 64-bit dHash (difference hash) for a grayscale image.
    :param np.ndarray image_gray: Grayscale image from video upload
    :return: 64-bit integer hash value representing the image
    :rtype: int
    """
    # Resize to 9x8 to get 8 rows and 9 columns
    # 9 columns produce 8 comparisons (0 vs 1, 1 vs 2, ..., 7 vs 8).
    # 8 rows with 8 bits per row = 64 bits
    resized = cv2.resize(image_gray, (9, 8), interpolation=cv2.INTER_AREA)
    # Compare horizontal neighbors
    diff = resized[:, 1:] > resized[:, :-1]
    # Pack bits into 64-bit integer
    bits = 0
    bit_index = 0
    for row in diff:
        for val in row:
            if bool(val):
                bits |= (1 << bit_index)
            bit_index += 1
    return int(bits)


def sample_frame_hashes(file_path: str, max_frames: int = 15) -> List[int]:
    """
    Sample frames across the video and compute dHashes for similarity.
    :param str file_path: Path to the video file to sample
    :param int max_frames: Maximum number of frames to sample across the video
    :return: List of 64-bit integer hash values for sampled frames
    :rtype: list[int]
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        cap.release()
        return []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count <= 0:
        frame_count = max_frames
    
    # Evenly sample frames across the video
    indices = np.linspace(0, max(0, frame_count - 1), num=min(max_frames, max(1, frame_count)), dtype=int)
    hashes = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hashes.append(dhash_gray(gray))
    
    cap.release()
    return hashes


def hamming_distance(a: int, b: int) -> int:
    """
    Calculate Hamming distance between two integers (number of differing bits).
    :param int a: First hash value to compare
    :param int b: Second hash value to compare
    :return: Number of differing bits between the two integers
    :rtype: int
    """
    return bin(a ^ b).count("1")


def compare_video_hashes(hashes_a: List[int], hashes_b: List[int]) -> float:
    """
    Compare two lists of video frame hashes and return a similarity score in [0, 1].
    1.0 means identical, 0.0 means completely different.
    :param List[int] hashes_a: Hash list for video A
    :param List[int] hashes_b: Hash list for video B
    :return: Similarity score between 0.0 and 1.0
    :rtype: float
    """
    if not hashes_a or not hashes_b:
        return 0.0
    
    # For each hash in set A, find the minimum distance to any hash in set B
    distances = []
    for h in hashes_a:
        min_dist = min(hamming_distance(h, hb) for hb in hashes_b)
        distances.append(min_dist)
    
    # Normalize by 64 bits (dHash produces 64-bit values)
    avg_distance = float(np.mean(distances))
    similarity = max(0.0, 1.0 - (avg_distance / 64.0))
    return similarity


def extract_video_metadata(file_path):
    """
    Extract the video metadata from the file.
    :param str file_path: The path to the video file.
    :return: The video metadata.
    :rtype: tuple
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise Exception("Could not open video file")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ratio_bucket = find_ratio_bucket(width, height)
    greatest_common_divisor = math.gcd(width, height)
    aspect_ratio = f"{int(width/greatest_common_divisor)}:{int(height/greatest_common_divisor)}"
    cap.release()
    return width, height, aspect_ratio, ratio_bucket

@app.post("/upload")
async def upload_videos(files: list[UploadFile] = File(...)):
    """
    Upload one or more MP4 videos and extract metadata
    :return: A JSONResponse of a list of video metadata dictionaries.
    :rtype: JSONResponse
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    results = []
    temp_paths = []
    try:
        for file in files:
            filename = file.filename or ""
            log.info(f"Processing file: {filename}")
            content_type = getattr(file, "content_type", None) or ""
            is_mp4_by_name = filename.lower().endswith('.mp4')
            is_mp4_by_type = content_type.lower() == 'video/mp4'
            if not (is_mp4_by_name or is_mp4_by_type):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Only MP4 files are supported. Received filename='{filename}', content_type='{content_type}'"
                )
            video_id = get_next_video_id()
            temp_path = f"temp_{video_id}.mp4"
            temp_paths.append(temp_path)
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            width, height, aspect_ratio, ratio_bucket = extract_video_metadata(temp_path)
            frame_hashes = sample_frame_hashes(temp_path)
            video_metadata = {
                "video_id": video_id,
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "ratio_bucket": ratio_bucket,
                "filename": filename
            }
            frame_hash_dict[video_id] = frame_hashes
            results.append(video_metadata)
            videos[video_id] = video_metadata
        return JSONResponse(content=results)
    except HTTPException:
        # Let HTTPExceptions propagate so their detail/status are preserved
        raise
    except Exception as e:
        log.error(f"Unexpected error processing videos: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing videos: {e}")
    finally:
        for p in temp_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    log.error(f"Error deleting temp file {p}")
                    pass


@app.get("/videos")
async def list_videos(ratio: Optional[str] = Query(default=None, description="Canonical aspect ratio like 9:16, 1:1, 4:5, 16:9")):
    """
    List uploaded videos, optionally filtered by canonical aspect ratio bucket.
    :param Optional[str] ratio: Canonical aspect ratio filter (i.e., 9:16, 1:1, 4:5, 16:9, Other)
    :return: List of uploaded video metadata dictionaries
    :rtype: list[dict]
    """
    values = list(videos.values())
    log.info(f"Listing videos with ratio: {ratio}")
    if ratio is None:
        return values
    matching_videos = []
    for video in values:
        log.info(f"Video: {video['filename']} has ratio: {video['ratio_bucket']}")
        if video["ratio_bucket"] == ratio:
            matching_videos.append(video)
    return matching_videos


@app.get("/match")
async def match_videos(video_id: str = Query(..., description="Video ID to match against other videos")):
    """
    Find videos with similar content to the given video_id.
    Uses perceptual hashing that's robust to overlays, aspect ratio, brightness, etc.
    :param str video_id: Video ID to match against other uploaded videos
    :return: List of similar videos with filename and confidence score
    :rtype: list[dict]
    """
    if video_id not in frame_hash_dict:
        raise HTTPException(status_code=404, detail=f"Video with ID {video_id} not found")
    
    target_hashes = frame_hash_dict[video_id]
    
    if not target_hashes:
        raise HTTPException(status_code=400, detail="Target video has no frame hashes")
    
    similar_videos = []
    
    for video_id_check, video_hashes in frame_hash_dict.items():
        if video_id_check == video_id:
            continue
        
        similarity = compare_video_hashes(target_hashes, video_hashes)
        
        if similarity > 0.6:
            similar_videos.append({
                "video_id": video_id_check,
                "filename": videos[video_id_check]["filename"],
                "confidence": round(similarity, 4)
            })
    
    # Sort by confidence (highest first)
    similar_videos.sort(key=lambda x: x["confidence"], reverse=True)
    
    return similar_videos


@app.get("/")
async def root():
    """
    Root endpoint to give description
    :return: Dictionary of description and endpoints
    :rtype: dict
    """
    log.info("Root endpoint called")
    return {
        "message": "Aspect Ratio Finder and Same Video Detector",
        "endpoints": {
            "upload": "POST /upload - Upload video files",
            "videos": "GET /videos?ratio=<ratio> - List all videos or filter by aspect ratio",
            "match": "GET /match?video_id=<id> - Find videos with similar content",
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)

