# Third-party components

- Reference implementation, original video and custom YOLO checkpoint: [Yingbo-Jiao/PoseVision](https://github.com/Yingbo-Jiao/PoseVision), commit `2d385598a26c187b2344b4f63b4ceee6ebdd2518`. The inspected repository does not provide a top-level LICENSE. Retain attribution and obtain appropriate permission before redistributing its code, weights or footage. Original upstream code, footage and checkpoints are not committed to this repository; acquisition scripts fetch the pinned originals from their source. Runtime model exports retain the original authorship and applicable model terms.
- YOLO11 architecture/export: [Ultralytics](https://github.com/ultralytics/ultralytics), AGPL-3.0 / commercial licensing. Raw output layout follows [Rockchip Model Zoo](https://github.com/airockchip/rknn_model_zoo) (Apache-2.0).
- RTMPose / MMPose / MMDetection / MMCV / MMEngine: [OpenMMLab](https://github.com/open-mmlab/mmpose), Apache-2.0. The model uses the original pretrained checkpoint, with no training or weight replacement.
- DeepSORT implementation and bundled ImageNet MobileNetV2 appearance checkpoint: [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime), MIT. The appearance model is exported unchanged to ONNX and FP16 RKNN; the optimized default runs appearance inference on RK3576 NPU, with an explicit ONNX CPU alternative.
- RKNN Toolkit2 / Lite2 and board runtime: [Rockchip](https://github.com/airockchip/rknn-toolkit2), subject to their distributed license terms.
- OpenCV (Apache-2.0), NumPy/SciPy (BSD), ONNX Runtime/FastAPI (MIT), and imageio-ffmpeg (BSD; its distributed FFmpeg binary has its own LGPL/GPL terms).

No blanket license is applied to third-party weights, video, or copied upstream source. Consult each component's current license before redistribution or commercial deployment.
