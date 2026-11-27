"""Write the optimization report only after the complete optimized run passes."""
import json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];d=ROOT/'docs'
read=lambda n:json.loads((d/n).read_text())
old=read('full-report.json');new=read('optimized-full-report.json');agreement=read('optimized-full-agreement.json');video=read('optimized-video-validation.json');summary=read('optimized-trace-summary.json');job=read('optimized-web-job.json')
assert old['frames']==new['frames']==video['decoded_frames']==1500
assert old['detection_npu_calls']==new['detection_npu_calls']==6000
ratio=new['end_to_end_fps']/old['end_to_end_fps']
text=f'''# RK3576 performance optimization

## Complete original-video comparison

Both runs process the unmodified 3840 × 2160, 50 FPS, 1500-frame `sample2.mp4`, with detection, three high-resolution ball tiles per frame, DeepSORT, jersey-color groups, per-confirmed-player hybrid pose, camera compensation, tactical court, JSONL and H.264 output enabled. There is no frame skipping, cached detection replay, interpolated skeleton, reduced detector resolution or disabled stage in the optimized run.

| Measurement | Original | Optimized |
| --- | --- | --- |
| Frames | {old['frames']} | {new['frames']} |
| Total wall time, seconds | {old['elapsed_seconds']:.2f} | {new['elapsed_seconds']:.2f} |
| End-to-end FPS | {old['end_to_end_fps']:.3f} | **{new['end_to_end_fps']:.3f}** |
| Detection NPU calls | {old['detection_npu_calls']} | {new['detection_npu_calls']} |
| Detection NPU calls/s | {old['detection_npu_fps']:.2f} | {new['detection_npu_fps']:.2f} |
| Detection frame-equivalent NPU FPS | {old['detection_npu_frame_fps']:.2f} | {new['detection_npu_frame_fps']:.2f} |
| Pose backbone NPU crops/s | {old['pose_npu_fps']:.2f} | {new['pose_npu_fps']:.2f} |
| ReID backend | CPU ONNX | FP16 RKNN NPU |
| ReID NPU crops/s | — | {new['reid_npu_fps']:.2f} |
| Pose observations | {old['pose_calls']} | {new['pose_calls']} |

Measured end-to-end speedup: **{ratio:.2f}×**. This remains an offline analytics pipeline; no real-time 25/50 FPS claim is made. NPU rates measure blocking RKNN API calls. The optimized combined NPU timing also includes ReID, which previously ran on CPU, so its denominator represents more work than the old combined NPU metric.

## Changes retained

1. Move unchanged pretrained MobileNetV2 appearance inference to FP16 RKNN. Real-crop feature parity versus ONNX CPU has minimum cosine 0.999836. Explicit CPU fallback is `BASKETBALL_REID=onnx`; its inputs are batched across classes once per frame with float32 in-place normalization.
2. Bound BLAS to one thread, OpenCV to two threads, and keep ONNX Runtime intra-op threads at two. This reduces synchronization on small Kalman/cosine operations. `BASKETBALL_CPU_THREADS` controls OpenCV threads.
3. Bound DeepSORT's descriptor gallery to 100 features per identity. IDs can differ after occlusion; agreement with previous IDs is not a ground-truth tracking metric.
4. Use both RK3576 NPU cores for detector and ReID, following same-input latency and identical-output tests. Keep automatic scheduling for pose, since dual-core execution was slower there. Overrides are `BASKETBALL_DETECTOR_CORES=0` and `BASKETBALL_REID_CORES=0`.
5. Apply the class confidence gate before DFL softmax/box decoding. Low-score grid cells cannot survive this gate, so computing their boxes was redundant. The score threshold, NMS policy and model weights are unchanged. Real-frame decoder checks have maximum coordinate difference below 0.001 original-image pixels, with identical scores/classes.
6. Prefetch decode and drain the encoder through ordered two-frame queues. Backpressure bounds memory; frames are never discarded. Eight real source frames matched sequential decode hashes exactly, and encoder drain preserved all frames. FFmpeg encoding is bounded to two threads.
7. Add stage wall timers, ReID NPU counters, model/core selection and queue timing to reports. Video worker times overlap main-thread work and must not be added to the main-thread stage sum.

## Profiling evidence

The initial same-window profile (source frames 500–599) took 134.00 seconds. Tracking/ReID consumed 67.66 seconds, detection 35.60 seconds and pose 14.18 seconds. An intermediate optimized profile, before the final score-first DFL change, took 75.83 seconds; tracking/ReID fell to 19.27 seconds. See `optimization-before.json`, `optimization-window-final.json` and `optimization-window-agreement.json`. The final full-video numbers above include the later DFL optimization and court compensation.

Final main-thread stage wall times (seconds):

```json
{json.dumps(new['stage_wall_seconds'],indent=2)}
```

Decode worker time: {new['decode_worker_seconds']:.2f} s; main-thread decode waiting: {new['decode_wait_seconds']:.2f} s; encoder worker write time: {new['encoder_worker_write_seconds']:.2f} s. These worker times overlap other work.

## Quality and integrity checks

The final full trace is compared with the previous real inference, independent of track numbering. Detection retention at IoU ≥ 0.5: `{json.dumps(agreement['retention_against_baseline'])}`; mean matched IoU: {agreement['mean_matched_detection_iou']:.8f}. Compared valid joints: {agreement['compared_keypoints']}; mean coordinate difference: {agreement['mean_keypoint_difference_pixels']:.4f} original-image pixels. This measures baseline agreement, not ground-truth AP/MOTA/PCK.

Ball-positive frames: {summary['frames_with_ball_detection']}. Mapping status counts: `{json.dumps(summary['mapping_status_frames'])}`. Output verification decoded all 1500 frames: H.264, 1360 × 540, 50 FPS, 30 seconds. Per-track confirmations and IDs can change because appearance inference precision and gallery history change; pose is still computed for every currently confirmed player.

## Rejected experiment and remaining limits

The INT8 detector candidate was faster (32.80 calls/s versus 16.97 FP16 in that held-out comparison), but retained only about 67.7% of FP16 player detections and 66.7% of referee detections. It failed the quality gate and is **not** enabled or shipped as a runtime model. The experiment used 128 real calibration images and no retraining; see `int8-evaluation.json`.

Full RTMPose RKNN still produces invalid outputs on this runtime; the verified NPU-backbone/CPU-head split remains. Hardware MPP/RGA video processing is **not enabled**: device nodes exist, but no usable MPP/RGA userspace codec stack was found in the inspected environment. This update keeps software video codecs and overlaps their work. Further acceleration would require separately validated codec integration, model batching or native runtime work; no unmeasured gain is promised.

## Reproduce and inspect

```bash
python -m pip install -r requirements.txt
python -m basketball.pipeline --output outputs/optimized-check --pose hybrid --calibration docs/sample2-calibration.json
# Optional explicit CPU appearance alternative:
BASKETBALL_REID=onnx python -m basketball.pipeline --output outputs/cpu-reid-check --pose hybrid
```

To rebuild the new appearance RKNN on the documented Linux conversion host: `python tools/export_reid.py` then `python tools/convert_rknn.py reid`. Full current metrics are `optimized-full-report.json`; integrity and agreement are `optimized-video-validation.json` and `optimized-full-agreement.json`.

[Open the optimized saved WebUI job]({job['url']}).

![Actual optimized output](optimized-result-preview.jpg)
'''
(d/'PERFORMANCE.md').write_text(text,encoding='utf-8');print(d/'PERFORMANCE.md')
