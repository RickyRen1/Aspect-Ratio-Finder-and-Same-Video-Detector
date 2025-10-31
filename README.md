# Aspect-Ratio-Finder-and-Same-Video-Detector
Simple API to upload videos and grab metadata and detect similar video based on content.

Public Render URL https://aspect-ratio-finder-and-same-video.onrender.com/

### Example Curl Commands
Uploading multiple videos
```bash
curl https://aspect-ratio-finder-and-same-video.onrender.com/upload -F "files=@test1.mp4" -F "files=@test2.mp4" -F "files=@test3.mp4"
```
Example Response
```json
[{"video_id":"85340911","width":1080,"height":1350,"aspect_ratio":"4:5","ratio_bucket":"4:5","filename":"test1.mp4"},{"video_id":"56059548","width":576,"height":576,"aspect_ratio":"1:1","ratio_bucket":"1:1","filename":"test2.mp4"},{"video_id":"65621593","width":576,"height":1024,"aspect_ratio":"9:16","ratio_bucket":"9:16","filename":"test3.mp4"}]
```
List All Videos
```bash
curl https://aspect-ratio-finder-and-same-video.onrender.com/videos
```
Example Response
```json
[{"video_id":"35909669","width":1080,"height":1350,"aspect_ratio":"4:5","ratio_bucket":"4:5","filename":"test1.mp4"},{"video_id":"00607680","width":576,"height":576,"aspect_ratio":"1:1","ratio_bucket":"1:1","filename":"test2.mp4"},{"video_id":"07602535","width":576,"height":1024,"aspect_ratio":"9:16","ratio_bucket":"9:16","filename":"test3.mp4"},{"video_id":"83039012","width":720,"height":1280,"aspect_ratio":"9:16","ratio_bucket":"9:16","filename":"youtube.mp4"},{"video_id":"64047974","width":1920,"height":1080,"aspect_ratio":"16:9","ratio_bucket":"16:9","filename":"billwurtz.mp4"}]
```
List Videos with Ratio
```bash
curl https://aspect-ratio-finder-and-same-video.onrender.com/videos?ratio=9:16
```
Example Response
```json
[{"video_id":"07602535","width":576,"height":1024,"aspect_ratio":"9:16","ratio_bucket":"9:16","filename":"test3.mp4"},{"video_id":"83039012","width":720,"height":1280,"aspect_ratio":"9:16","ratio_bucket":"9:16","filename":"youtube.mp4"}]
```
List Videos with Matching Content
```bash
curl https://aspect-ratio-finder-and-same-video.onrender.com/match?video_id=64047974
```
Example Response
```json
[{"video_id":"80821672","filename":"test2.mp4","confidence":1.0},{"video_id":"51926797","filename":"test3_overlay.mp4","confidence":0.9642},{"video_id":"72660832","filename":"test3.mp4","confidence":0.9627}]
```

### Design Notes
- Content Similarity: The service uses a two-stage matching algorithm for robust video similarity detection:
  1. **Stage 1 (Hash-based filtering)**: Computes 64-bit pHash (DCT-based perceptual hash) values on up to 25 evenly sampled frames per video. Frames are center-cropped to square format and resized to reduce aspect ratio sensitivity. Videos with hash similarity > 0.50 proceed to stage 2.
  2. **Stage 2 (Keypoint matching)**: Uses ORB (Oriented FAST and Rotated BRIEF) keypoint detection and matching on frame thumbnails (128×128 pixels) to verify matches. Keypoint matching is more discriminative and helps distinguish true matches from false positives.
  3. **Combined scoring**: Final similarity combines hash similarity (30% weight) and keypoint similarity (70% weight). True matches (same video, different aspect ratio) should score > 0.85.

- Aspect Ratio Bucketing: Videos are placed into canonical buckets 9:16, 1:1, 4:5, 16:9 within 1% tolerance. If the video does not fit in any bucket then they are placed in `Other`

- In-memory state: Uploaded video metadata, frame hashes, and frame thumbnails for similarity detection are stored in-memory (`videos`, `frame_hash_dict`, `frame_thumbnails_dict`) so restarts will clear state. In order to process video metadata, uploads are written to temporary files which may take more than a minute depending on internet speed.

- Limits: Only `.mp4` is accepted. Videos are sampled at up to 25 frames evenly distributed across the video duration for similarity comparison.
