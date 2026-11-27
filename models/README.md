# Runtime models

The repository includes these final runtime assets:

- `detector.rknn`: custom four-class YOLO11n, FP16, RK3576.
- `pose_backbone.rknn` + `pose_head.onnx`: verified hybrid RTMPose-s.
- `pose.onnx`: explicit full CPU pose alternative and numerical reference.
- `reid.rknn`: verified FP16 pretrained MobileNetV2 for DeepSORT appearance inference on NPU.
- `reid.onnx`: explicit CPU ReID alternative (`BASKETBALL_REID=onnx`) and numerical reference.

JSON files record normalization, original hashes, build times and the failed full-pose experiment. The invalid full-pose RKNN binary and intermediate detector/backbone ONNX exports are excluded. Follow the root README to regenerate them for conversion or parity diagnostics. Original checkpoints and sample footage are downloaded from their upstream sources, not committed here.

The experimental INT8 detector was rejected after held-out comparisons and is not a shipped runtime asset. Details are in `docs/int8-evaluation.json`.
