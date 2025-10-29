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
[{"video_id":"00607680","filename":"test2.mp4","confidence":0.6573},{"video_id":"07602535","filename":"test3.mp4","confidence":0.6219}]
```

### Design Notes
- Content Similarity: The service computes 64-bit dHash values on evenly sampled grayscale frames across the video, then compares sets of hashes with a Hamming-distance–based similarity score in [0, 1]. We return the videos that have a similarity rating greater than 0.6, sorted most to least similar. 0.6 was decided after testing a couple videos so a different value may be more robust.

- Aspect Ratio Bucketing: Videos are placed into canonical buckets 9:16, 1:1, 4:5, 16:9 within 1% tolerance. If the video does not fit in any bucket then they are placed in `Other`

- In-memory state: Uploaded video metadata and frame hashes for checking similar content are stored in-memory (`videos`, `frame_hash_dict`) so restarts will clear state. In order to process video metadata, uploads are written to temporary files which may take more than a minute depending on internet speed.

- Limits: Only `.mp4` is accepted. Very long videos will probably mean fewer sampled frames compared to total video causing similarity to be lower.
