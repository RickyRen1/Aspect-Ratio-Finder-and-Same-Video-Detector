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

# Storage for frame thumbnails (center-cropped square grayscale images) for keypoint matching
# video_id: list of numpy arrays (grayscale frames)
frame_thumbnails_dict = {}

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


def phash_gray(image_gray: np.ndarray) -> int:
    """
    Compute a 64-bit pHash (DCT-based perceptual hash) for a grayscale image.
    More robust to scaling/brightness and minor crops than dHash.
    :param np.ndarray image_gray: Grayscale image
    :return: 64-bit integer hash value representing the image
    :rtype: int
    """
    # Resize to 32x32, compute DCT, then take top-left 8x8 (excluding DC)
    resized = cv2.resize(image_gray, (32, 32), interpolation=cv2.INTER_AREA)
    resized = resized.astype(np.float32)
    dct = cv2.dct(resized)
    dct_low = dct[:8, :8].copy()
    # Exclude the DC component (0,0) from median calculation
    dct_flat = dct_low.flatten()
    dct_no_dc = dct_flat[1:]  # Skip DC component at index 0
    median_val = np.median(dct_no_dc)
    bits = 0
    bit_index = 0
    for r in range(8):
        for c in range(8):
            coeff = dct_low[r, c]
            # Skip DC position but keep bit index consistent at 64 bits
            bit = 1 if coeff > median_val else 0
            if bit:
                bits |= (1 << bit_index)
            bit_index += 1
    return int(bits)


def _detect_content_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    """
    Estimate content bounding box by removing low-texture side/top/bottom bars (black or blurred).
    Uses gradient energy to find active columns/rows, robust to brightness.
    Returns (y0, y1, x0, x1) inclusive-exclusive bounds.
    """
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return 0, h, 0, w
    # Gradient magnitude approximation
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # Aggregate per column and row
    col_energy = np.mean(mag, axis=0)
    row_energy = np.mean(mag, axis=1)
    # Normalize
    col_max = float(np.max(col_energy)) or 1.0
    row_max = float(np.max(row_energy)) or 1.0
    col_norm = col_energy / col_max
    row_norm = row_energy / row_max
    # Threshold relative to max to keep active content; ignore quiet borders (black/blurred)
    # Use a low threshold to keep real content even if low-contrast
    col_thresh = 0.15
    row_thresh = 0.15
    active_cols = col_norm > col_thresh
    active_rows = row_norm > row_thresh
    # Find outermost active indices with a minimum interior size
    def _active_bounds(active: np.ndarray, min_len: int) -> tuple[int, int]:
        if not np.any(active):
            return 0, len(active)
        idx = np.where(active)[0]
        start = int(idx[0])
        end = int(idx[-1]) + 1
        if end - start < min_len:
            # Expand to minimum length centered
            center = (start + end) // 2
            half = min_len // 2
            start = max(0, center - half)
            end = min(len(active), start + min_len)
            start = max(0, end - min_len)
        return start, end
    min_w = max(32, int(0.4 * w))
    min_h = max(32, int(0.4 * h))
    x0, x1 = _active_bounds(active_cols, min_w)
    y0, y1 = _active_bounds(active_rows, min_h)
    return y0, y1, x0, x1


def _content_square_grayscale(frame: np.ndarray, output_size: int = 64) -> np.ndarray:
    """
    Convert BGR frame to grayscale, detect content bbox (removing bars/blur borders),
    then take a centered square crop of the content region and resize to output_size.
    """
    if frame is None or frame.size == 0:
        return np.zeros((output_size, output_size), dtype=np.uint8)
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    y0, y1, x0, x1 = _detect_content_bbox(gray_full)
    y0 = max(0, min(y0, gray_full.shape[0]))
    y1 = max(y0 + 1, min(y1, gray_full.shape[0]))
    x0 = max(0, min(x0, gray_full.shape[1]))
    x1 = max(x0 + 1, min(x1, gray_full.shape[1]))
    content = gray_full[y0:y1, x0:x1]
    ch, cw = content.shape[:2]
    side = min(ch, cw)
    cy0 = (ch - side) // 2
    cx0 = (cw - side) // 2
    square = content[cy0:cy0+side, cx0:cx0+side]
    if square.size == 0:
        square = gray_full
    square_resized = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return square_resized


def sample_frame_hashes(file_path: str, max_frames: int = 25) -> tuple[List[int], List[np.ndarray]]:
    """
    Sample frames across the video and compute pHashes for similarity.
    Also returns frame thumbnails for keypoint matching.
    :param str file_path: Path to the video file to sample
    :param int max_frames: Maximum number of frames to sample across the video
    :return: Tuple of (list of 64-bit integer hash values, list of grayscale frame thumbnails)
    :rtype: tuple[list[int], list[np.ndarray]]
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        cap.release()
        return [], []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count <= 0:
        frame_count = max_frames
    
    # Evenly sample frames across the video
    indices = np.linspace(0, max(0, frame_count - 1), num=min(max_frames, max(1, frame_count)), dtype=int)
    hashes = []
    thumbnails = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # Use content-aware crop to remove bars/blur borders and focus on content
        gray_sq_large = _content_square_grayscale(frame, output_size=128)
        # Use smaller size for hashing (64x64)
        gray_sq_small = cv2.resize(gray_sq_large, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Compute hash
        hashes.append(phash_gray(gray_sq_small))
        
        # Store thumbnail for keypoint matching
        thumbnails.append(gray_sq_large)
    
    cap.release()
    return hashes, thumbnails


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
    
    # Bidirectional nearest-neighbor distances with percentile aggregation
    def _min_dists(src: List[int], dst: List[int]) -> List[int]:
        out = []
        for h in src:
            out.append(min(hamming_distance(h, hb) for hb in dst))
        return out

    dists_ab = _min_dists(hashes_a, hashes_b)
    dists_ba = _min_dists(hashes_b, hashes_a)

    if not dists_ab or not dists_ba:
        return 0.0

    # Use 20th percentile for balance: tolerant enough for overlays, strict enough to reject false positives
    # Overlays can cause some frames to have higher distances, but true matches should still have many low-distance matches
    perc_ab = float(np.percentile(dists_ab, 20))
    perc_ba = float(np.percentile(dists_ba, 20))
    avg_distance = (perc_ab + perc_ba) / 2.0
    similarity = max(0.0, 1.0 - (avg_distance / 64.0))
    return similarity


def compare_frames_keypoints(frames_a: List[np.ndarray], frames_b: List[np.ndarray]) -> float:
    """
    Compare two lists of frames using keypoint matching (ORB detector + BF matcher).
    Uses geometric verification (homography/RANSAC) to filter overlay matches.
    Returns a similarity score in [0, 1] based on geometrically consistent matches.
    :param List[np.ndarray] frames_a: List of grayscale frames from video A
    :param List[np.ndarray] frames_b: List of grayscale frames from video B
    :return: Similarity score between 0.0 and 1.0
    :rtype: float
    """
    if not frames_a or not frames_b:
        return 0.0
    
    # Initialize ORB detector (more features for better overlay tolerance)
    orb = cv2.ORB_create(nfeatures=1000)
    # Brute force matcher with Hamming distance for ORB descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    all_match_ratios = []
    
    # Compare each frame from A with all frames from B, take best match
    for frame_a in frames_a:
        h_a, w_a = frame_a.shape[:2]
        # Create mask to ignore border regions (overlays often appear on edges)
        # Keep central 80% of image
        border_x = int(0.1 * w_a)
        border_y = int(0.1 * h_a)
        mask_a = np.zeros((h_a, w_a), dtype=np.uint8)
        mask_a[border_y:h_a-border_y, border_x:w_a-border_x] = 255
        
        kp_a, desc_a = orb.detectAndCompute(frame_a, mask_a)
        if desc_a is None or len(kp_a) < 10:  # Need sufficient keypoints
            continue
        
        best_ratio = 0.0
        for frame_b in frames_b:
            h_b, w_b = frame_b.shape[:2]
            # Similar mask for frame B
            border_x_b = int(0.1 * w_b)
            border_y_b = int(0.1 * h_b)
            mask_b = np.zeros((h_b, w_b), dtype=np.uint8)
            mask_b[border_y_b:h_b-border_y_b, border_x_b:w_b-border_x_b] = 255
            
            kp_b, desc_b = orb.detectAndCompute(frame_b, mask_b)
            if desc_b is None or len(kp_b) < 10:
                continue
            
            # Match descriptors using KNN (k=2 for Lowe's ratio test)
            try:
                matches = bf.knnMatch(desc_a, desc_b, k=2)
            except cv2.error:
                continue
            
            # Apply Lowe's ratio test to filter good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                        good_matches.append(m)
            
            if len(good_matches) < 10:
                continue
            
            # Use geometric verification (homography with RANSAC) to filter overlay matches
            # Overlay matches won't be geometrically consistent with content matches
            src_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            try:
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0)
                if mask is not None:
                    inlier_count = int(mask.sum())
                    # Use inlier ratio instead of raw match ratio (filters overlay matches)
                    match_ratio = inlier_count / max(len(kp_a), 1)
                else:
                    # Fallback to raw match ratio if homography fails
                    match_ratio = len(good_matches) / max(len(kp_a), 1)
            except cv2.error:
                # Fallback to raw match ratio if homography fails
                match_ratio = len(good_matches) / max(len(kp_a), 1)
            
            best_ratio = max(best_ratio, match_ratio)
        
        if best_ratio > 0:
            all_match_ratios.append(best_ratio)
    
    if not all_match_ratios:
        return 0.0
    
    # Use median match ratio as similarity score, scale to [0, 1]
    # For similar videos with overlays, geometric verification ensures consistent matches
    median_ratio = float(np.median(all_match_ratios))
    # Normalize: 0.15+ matches = high similarity (more tolerant for overlays)
    # This ensures true matches (even with overlays) score well
    similarity = min(1.0, median_ratio / 0.15)
    return similarity


def compute_combined_similarity(hash_similarity: float, keypoint_similarity: float, hash_weight: float = 0.4, keypoint_weight: float = 0.6) -> float:
    """
    Combine hash-based and keypoint-based similarity scores.
    Keypoints are weighted more heavily (60%) as they're more discriminative.
    Calibrated to ensure true matches (even with overlays) score > 0.85.
    :param float hash_similarity: Similarity from perceptual hashing [0, 1]
    :param float keypoint_similarity: Similarity from keypoint matching [0, 1]
    :param float hash_weight: Weight for hash similarity (default 0.4)
    :param float keypoint_weight: Weight for keypoint similarity (default 0.6)
    :return: Combined similarity score [0, 1]
    :rtype: float
    """
    total_weight = hash_weight + keypoint_weight
    combined = (hash_similarity * hash_weight + keypoint_similarity * keypoint_weight) / total_weight
    # Boost combined score for true matches (overlays may reduce individual scores slightly)
    # More tolerant thresholds for overlay scenarios
    if hash_similarity > 0.55 and keypoint_similarity > 0.45:
        combined = min(1.0, combined * 1.15)  # 15% boost for moderate-strong matches (handles overlays)
    elif hash_similarity > 0.50 and keypoint_similarity > 0.40:
        combined = min(1.0, combined * 1.08)  # 8% boost for lower scores that might be overlay cases
    return max(0.0, min(1.0, combined))


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
            frame_hashes, frame_thumbnails = sample_frame_hashes(temp_path)
            video_metadata = {
                "video_id": video_id,
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "ratio_bucket": ratio_bucket,
                "filename": filename
            }
            frame_hash_dict[video_id] = frame_hashes
            frame_thumbnails_dict[video_id] = frame_thumbnails
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
    Uses two-stage matching: perceptual hashing (stage 1) and keypoint matching (stage 2).
    :param str video_id: Video ID to match against other uploaded videos
    :return: List of similar videos with filename and confidence score
    :rtype: list[dict]
    """
    target_confidence = 0.8 # Threshold for the target video to be considered similar
    if video_id not in frame_hash_dict:
        raise HTTPException(status_code=404, detail=f"Video with ID {video_id} not found")
    
    target_hashes = frame_hash_dict[video_id]
    target_thumbnails = frame_thumbnails_dict.get(video_id, [])
    
    if not target_hashes:
        raise HTTPException(status_code=400, detail="Target video has no frame hashes")
    
    similar_videos = []
    
    # Stage 1: Hash-based filtering (fast, catch potential matches)
    hash_candidates = []
    for video_id_check, video_hashes in frame_hash_dict.items():
        if video_id_check == video_id:
            continue
        
        hash_similarity = compare_video_hashes(target_hashes, video_hashes)
        
        # Lower threshold for stage 1 (catch more candidates for stage 2 verification)
        # Need to be more permissive here to catch all potential matches
        if hash_similarity > 0.50:
            hash_candidates.append({
                "video_id": video_id_check,
                "hash_similarity": hash_similarity
            })
    
    # Stage 2: Keypoint matching for candidates (more discriminative)
    for candidate in hash_candidates:
        video_id_check = candidate["video_id"]
        hash_sim = candidate["hash_similarity"]
        
        # Get thumbnails for keypoint matching
        candidate_thumbnails = frame_thumbnails_dict.get(video_id_check, [])
        
        if target_thumbnails and candidate_thumbnails:
            # Perform keypoint matching
            keypoint_sim = compare_frames_keypoints(target_thumbnails, candidate_thumbnails)

            # If the keypoint similarity is between 0 and target_confidence, check the reversed keypoint similarity
            # to see if the number of keypoints affects the similarity score
            if keypoint_sim > 0.0 and keypoint_sim < target_confidence:
                keypoint_sim = max(keypoint_sim, compare_frames_keypoints(candidate_thumbnails, target_thumbnails))
            # Combine hash and keypoint similarity scores
            combined_similarity = compute_combined_similarity(hash_sim, keypoint_sim)
        else:
            # Fallback to hash-only if thumbnails unavailable
            combined_similarity = hash_sim * (target_confidence + 0.05)  # Slight penalty for no keypoint check
        
        if combined_similarity > target_confidence:
            similar_videos.append({
                "video_id": video_id_check,
                "filename": videos[video_id_check]["filename"],
                "confidence": round(combined_similarity, 4)
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
