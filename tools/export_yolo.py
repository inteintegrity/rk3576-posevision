"""Export original PoseVision YOLO11n unchanged weights with a raw three-scale head.

The raw head contract follows Rockchip RKNN Model Zoo YOLOv8 (Apache-2.0).
DFL decoding and class-aware NMS execute on CPU. No training is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
import onnx
from ultralytics import YOLO


def raw_head(self, x):
    outputs = []
    for i in range(self.nl):
        outputs.append(self.cv2[i](x[i]))
        scores = self.cv3[i](x[i]).sigmoid()
        outputs.extend((scores, scores.sum(1, keepdim=True).clamp(0, 1)))
    return outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='upstream/yolo_detection/best.pt')
    p.add_argument('--output', default='models/detector.onnx')
    a = p.parse_args()
    torch.set_num_threads(4)
    yolo = YOLO(a.weights)
    model = yolo.model.cpu().float().eval().fuse()
    head = model.model[-1]
    if head.nl != 3 or head.reg_max != 16:
        raise ValueError('Expected anchor-free YOLO with three scales and reg_max=16')
    head.forward = MethodType(raw_head, head)
    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        shapes = [list(x.shape) for x in model(dummy)]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(out), opset_version=12, dynamo=False,
                      input_names=['images'], output_names=[f'{k}_{s}'
                      for s in (8, 16, 32) for k in ('boxes', 'classes', 'score_sum')])
    onnx.checker.check_model(onnx.load(str(out)))
    out.with_suffix('.json').write_text(json.dumps({
        'weights_sha256': hashlib.sha256(Path(a.weights).read_bytes()).hexdigest(),
        'classes': yolo.names, 'input': [1, 3, 640, 640], 'outputs': shapes,
        'normalization': 'RGB / 255', 'letterbox_padding': 114,
        'trained': False}, indent=2))
    print(out, shapes)


if __name__ == '__main__':
    main()
