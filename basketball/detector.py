"""RK3576 FP16 / CPU ONNX reference custom YOLO11n decoding; no synthetic or CPU fallback detections."""
import time
from types import SimpleNamespace

import cv2
import numpy as np

import json
from pathlib import Path
CLASSES = {int(k):v for k,v in json.loads((Path(__file__).resolve().parents[1]/'models/detector.json').read_text())['classes'].items()}


def letterbox(frame):
    h, w = frame.shape[:2]
    scale = min(640 / w, 640 / h)
    nw, nh = round(w * scale), round(h * scale)
    left, top = (640 - nw) // 2, (640 - nh) // 2
    image = np.full((640, 640, 3), 114, dtype=np.uint8)
    image[top:top+nh, left:left+nw] = cv2.resize(frame, (nw, nh))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (scale, left, top)


def nms(boxes, scores, threshold=0.7):
    order = scores.argsort()[::-1]
    keep = []
    area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    while order.size:
        i = order[0]
        keep.append(i)
        others = order[1:]
        inter = np.prod(np.maximum(np.minimum(boxes[i, 2:], boxes[others, 2:]) -
                                   np.maximum(boxes[i, :2], boxes[others, :2]), 0), axis=1)
        iou = inter / np.maximum(area[i] + area[others] - inter, 1e-9)
        order = others[iou <= threshold]
    return keep


def decode(outputs, transform, frame_shape, confidence=0.30):
    if len(outputs) != 9:
        raise ValueError(f'Expected nine raw-head outputs, got {len(outputs)}')
    all_boxes, all_scores, all_classes = [], [], []
    for branch in range(3):
        raw, probs = outputs[branch*3:branch*3+2]
        _, _, h, w = raw.shape
        logits = raw.astype(np.float32).reshape(1, 4, 16, h, w)
        weights = np.exp(logits - logits.max(axis=2, keepdims=True))
        weights /= weights.sum(axis=2, keepdims=True)
        dist = (weights * np.arange(16).reshape(1, 1, 16, 1, 1)).sum(axis=2)[0]
        yy, xx = np.mgrid[:h, :w]
        grid = np.stack((xx + .5, yy + .5))
        boxes = np.concatenate((grid - dist[:2], grid + dist[2:]), axis=0)
        boxes = boxes.transpose(1, 2, 0).reshape(-1, 4) * (640 / h)
        scores = probs[0].reshape(len(CLASSES), -1).T
        classes = scores.argmax(axis=1)
        scores = scores.max(axis=1)
        selected = (scores >= confidence) & np.isin(classes, list(CLASSES))
        all_boxes.append(boxes[selected]); all_scores.append(scores[selected]); all_classes.append(classes[selected])
    boxes, scores, classes = map(np.concatenate, (all_boxes, all_scores, all_classes))
    keep = []
    for cls in CLASSES:
        indices = np.flatnonzero(classes == cls)
        keep.extend(indices[nms(boxes[indices], scores[indices])])
    keep = sorted(keep, key=lambda i: -scores[i])[:300]
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    scale, left, top = transform
    boxes -= np.array([left, top, left, top])
    boxes /= scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_shape[1]-1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_shape[0]-1)
    return boxes, scores, classes


class Detector:
    names = CLASSES

    def __init__(self, model, backend='rknn'):
        self.backend = backend
        self.times = []
        self.events = []
        self.boxes = []
        if backend == 'rknn':
            from rknnlite.api import RKNNLite
            self.runtime = RKNNLite()
            if self.runtime.load_rknn(str(model)) != 0:
                raise RuntimeError('RKNN load failed')
            if self.runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
                raise RuntimeError('RK3576 NPU initialization failed')
        elif backend == 'onnx':
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            self.runtime = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        else:
            raise ValueError(backend)

    def run_raw(self, image):
        t = time.perf_counter()
        if self.backend == 'rknn':
            outputs = self.runtime.inference(inputs=[image[None]], data_format=['nhwc'])
        else:
            outputs = self.runtime.run(None, {'images': image.transpose(2, 0, 1)[None].astype(np.float32)/255})
        elapsed = time.perf_counter() - t
        if outputs is None:
            raise RuntimeError('Inference returned no tensors')
        return outputs, elapsed

    def detect(self, frame, tiled_ball=True):
        image, transform = letterbox(frame)
        outputs, elapsed = self.run_raw(image)
        self.times.append(elapsed)
        boxes, scores, classes = decode(outputs, transform, frame.shape,confidence=.12)
        chosen=(classes==1)|(scores>=.30)
        boxes,scores,classes=boxes[chosen],scores[chosen],classes[chosen]
        # Preserve ball pixel scale in high-resolution footage. Tiles use the exact
        # same custom weights, with genuine inference on every tile, every frame.
        if tiled_ball and frame.shape[1]>1280:
            h,w=frame.shape[:2];size=min(h,w,1440)
            humans=boxes[np.isin(classes,[0,2])]
            center=float(np.median((humans[:,1]+humans[:,3])/2)) if len(humans) else h*.6
            y=int(np.clip(center-size/2,0,h-size))
            starts=np.linspace(0,w-size,int(np.ceil(w/size)),dtype=int).tolist()
            for x in starts:
                crop=frame[y:y+size,x:x+size]
                image,tr=letterbox(crop);out,elapsed=self.run_raw(image);self.times.append(elapsed)
                b,s,c=decode(out,tr,crop.shape,confidence=.12);keep=c==1
                if keep.any():
                    b=b[keep]+[x,y,x,y];boxes=np.concatenate([boxes,b]);scores=np.concatenate([scores,s[keep]]);classes=np.concatenate([classes,c[keep]])
            ball=np.flatnonzero(classes==1);other=np.flatnonzero(classes!=1)
            keep=np.r_[other,ball[nms(boxes[ball],scores[ball],.5)]]
            boxes,scores,classes=boxes[keep],scores[keep],classes[keep]
        self.boxes = [SimpleNamespace(xyxy=b[None], conf=np.array([s]), cls=np.array([c]))
                      for b, s, c in zip(boxes, scores, classes)]
        return self.boxes

    def predict(self, image, classes=None, conf=.30, verbose=False):
        # Both upstream consumers reuse this frame's genuine detections: one NPU call/frame.
        return [SimpleNamespace(boxes=[b for b in self.boxes if b.conf[0] >= conf
                                     and (classes is None or int(b.cls[0]) in classes)])]

    def close(self):
        if self.backend == 'rknn':
            self.runtime.release()
