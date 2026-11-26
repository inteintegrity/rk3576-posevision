import json,pathlib,hashlib
r=pathlib.Path(__file__).resolve().parents[1];docs=r/'docs'
full=json.loads((docs/'full-report.json').read_text());trace=json.loads((docs/'full-trace-summary.json').read_text());parity=json.loads((docs/'model_parity_hybrid.json').read_text());web=json.loads((docs/'completed-web-validation.json').read_text())
assert full['frames']==1500 and web['state']=='complete'
text=f'''# Device validation — 26 November 2026

Verified on Seeed Studio reComputer RK3576: ARM64, 8 GB RAM, Linux 6.1.115, Python 3.11.2, RKNN Toolkit2/Lite2 2.3.2, vendor librknnrt 2.3.0, kernel NPU driver 0.9.8. NPU frequency was observed at 950 MHz. Imported WebUI versions were FastAPI 0.115.12 and Uvicorn 0.34.2; the reused `.deps` directory retains older distribution metadata, so imported versions are recorded separately in `environment-final.json`.

## Original assets

- Upstream commit: `2d385598a26c187b2344b4f63b4ceee6ebdd2518`.
- `input/sample2.mp4`: unchanged, 3840 × 2160, 50 FPS, 1500 frames / 30 seconds, 95,708,286 bytes. Git blob SHA-1: `6e47c96313a27f3d0e7268d402c69646fc92cb4b`; SHA-256: `f05b16dc159e627cfaee6641167cbe4609a31c53a24e7b298960d17509fbff0a`.
- Original YOLO custom weight SHA-256: `fde2d88f0e51d62bff61b3176f5bf4c76ec6001523d581365de8bdc9d6527eb6`.
- Original RTMPose-s `fcb2599b_20230126` checkpoint, strict state-dictionary loading, no training.

## Compatibility and conversion parity

YOLO11n FP16 RKNN works on the device. Its first-frame regression tensor cosine similarities exceed 0.99998 against ONNX. The parity script additionally verifies decoded detection IoU and confidence differences on the same input. Near-zero class probabilities and the unused score-sum output have lower FP16 cosine; all nine values are retained in the JSON evidence.

The full RTMPose RKNN compiled and initialized but returned two entirely zero SimCC tensors. The first attempted validation failed with mean keypoint error 260.39 pixels. The exact failing operator was not isolated. It is an invalid deployment artifact, retained only for reproducibility with an explicit failed validation manifest.

The working alternative exports the original CSPNeXt backbone to RKNN and executes the unchanged RTMCC/SimCC head with ONNX Runtime CPU. First-frame mean valid-keypoint error versus the full original ONNX model: **{parity['pose_mean_keypoint_error_px']:.2f} pixels**; maximum **{parity['pose_max_keypoint_error_px']:.2f} pixels**, {parity['pose_checked_keypoints']} confidence-qualified joints. This is a small conversion check, not an accuracy evaluation over the dataset. The CPU pose option remains available explicitly. No fallback generates synthetic coordinates.

Build-only times were 2.56 seconds for YOLO, 2.10 seconds for experimental full pose, and 1.35 seconds for the working pose backbone; downloads and ONNX export are excluded. Hashes and exact measured times are in `models/*.build.json`.

## Staged validation

The first 100 original frames were processed with full-frame YOLO and DeepSORT: detection NPU 18.86 calls/s; end-to-end 3.20 frames/s. A subsequent 100-frame run added hybrid pose and floor camera registration: detection 19.61 calls/s, pose backbone 89.38 crops/s, end-to-end 2.45 frames/s. These stages did not yet include high-resolution ball tiles.

Full-frame reduction to 640 pixels missed the small ball. The same checkpoint on original-pixel tiles recovered real ball detections; the actual RK3576 NPU check on frame 600 detected a ball at confidence 0.625. The final pipeline uses one full-frame call plus three ball-tile calls per sample frame, threshold 0.12 for the ball and 0.30 for the other classes. Tiles are genuine inference, not interpolated ball positions. The lower threshold can also yield false positives.

## Final complete original-video run

All **{full['frames']} frames** completed with **no frame skipping**, hybrid RTMPose, NPU ball tiling, DeepSORT, team colors, and calibrated floor-motion compensation.

| Measurement | Actual result |
| --- | --- |
| Detection NPU inference calls | {full['detection_npu_calls']} |
| Detection NPU calls/s | {full['detection_npu_fps']:.2f} |
| Detection frame-equivalent NPU FPS | {full['detection_npu_frame_fps']:.2f} |
| Pose backbone NPU crops/s | {full['pose_npu_fps']:.2f} |
| Combined detection + pose NPU frame-equivalent FPS | {full['npu_pipeline_fps']:.2f} |
| End-to-end video frames/s | **{full['end_to_end_fps']:.2f}** |
| Total processing time | {full['elapsed_seconds']:.2f} seconds |
| Pose crops processed | {full['pose_calls']} |
| CPU pose-head time | {full['pose_cpu_head_seconds']:.2f} seconds |
| Frames with ball detections | {trace['frames_with_ball_detection']} |
| Mapped track observations | {trace['mapped_track_observations']} |

Mapping status counts: `{json.dumps(trace['mapping_status_frames'])}`. Team observation counts: `{json.dumps(trace['team_observations'])}`. Unknown labels remain unknown until enough appearance samples are available. Detections and team counts are observations, not verified accuracy, recall, MOTA, player identities, or jersey-number recognition.

NPU rates time blocking RKNN API calls only. End-to-end includes initialization, original 4K decoding, all preprocessing, CPU ReID/DeepSORT/teams, CPU pose head, floor camera registration, rendering, per-frame JSON, MP4 encoding and encoder flush. Upload is excluded. The MP4 timeline stays at source 50 FPS and is not the inference speed.

## Web and artifact verification

Integration tests exercised English page serving, original-sample selection, 95 MB MP4 upload, first-frame extraction, degenerate calibration rejection, single-job locking (HTTP 409), and withholding incomplete outputs. The completed result returned HTTP 206 for a byte-range video request. All 1500 JSONL records were parsed and verified to include real ball detections, 17-joint poses and calibrated positions. See `completed-web-validation.json`.

Browser automation opened the page but repeatedly timed out when acquiring its rendered state. Visual QA therefore uses the actual rendered inference frames and source-layout review; browser clicks and cross-browser playback are not claimed as automatically verified. The downloaded MP4 was decoded frame by frame: H.264, 1360 × 540, exactly 1500 frames at 50 FPS / 30 seconds, 25,762,328 bytes. The checksum and full results are in `video-validation.json`.

Five focused geometry/NMS tests passed, including rejecting degenerate calibration and withholding mapping after camera registration is lost. Python compilation and JavaScript syntax checks passed. Linux pinned-environment YOLO export was rerun successfully, and RTMPose backbone/head exports were used on the board.

## Inspect the result

- Reopen saved job: [English dashboard](http://192.168.3.120:8093/?job={web['job_id']}).
- [Processed MP4](http://192.168.3.120:8093/api/jobs/{web['job_id']}/result.mp4).
- Raw measurements: `full-report.json`, `full-trace-summary.json`, `model_parity_hybrid.json`, `model_parity.json`, `environment-final.json`.
- The first-frame paint-corner calibration is approximate and assumes a 28 × 15 m court with 5.8 × 4.9 m paint. The camera registration can drift. Loss of reliable registration suppresses positional dots rather than freezing a stale homography. Ball projection assumes floor contact and is approximate for airborne balls.

![Real final inference frame](result-preview.jpg)
'''
(docs/'VALIDATION.md').write_text(text,encoding='utf-8');print(docs/'VALIDATION.md')
