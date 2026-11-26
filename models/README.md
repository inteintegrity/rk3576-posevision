# Runtime models

The repository includes these final runtime assets:

- `detector.rknn`: custom four-class YOLO11n, FP16, RK3576.
- `pose_backbone.rknn` + `pose_head.onnx`: verified hybrid RTMPose-s.
- `pose.onnx`: explicit full CPU pose alternative and numerical reference.
- `reid.onnx`: pretrained MobileNetV2 appearance features for DeepSORT CPU inference.

JSON files record normalization, original hashes, build times and the failed full-pose experiment. The invalid full-pose RKNN binary and intermediate detector/backbone ONNX exports are excluded. Follow the root README to regenerate them for conversion or parity diagnostics. Original checkpoints and sample footage are downloaded from their upstream sources, not committed here.
