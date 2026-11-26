"""Compare real RKNN inference to ONNX CPU using original sample frames."""
import json, pathlib, sys, cv2, numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basketball.detector import Detector,letterbox
from basketball.pose import Pose
def cosine(a,b):
    a=a.astype(np.float64).ravel();b=b.astype(np.float64).ravel();return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
cap=cv2.VideoCapture(str(ROOT/'upstream/input/sample2.mp4'));ok,frame=cap.read();cap.release();assert ok
det=Detector(ROOT/'models/detector.rknn');ref=Detector(ROOT/'models/detector.onnx','onnx')
image,transform=letterbox(frame);a,_=det.run_raw(image);b,_=ref.run_raw(image)
result={'detector_tensor_cosines':[cosine(x,y) for x,y in zip(a,b)],'rknn_detections':[]}
boxes=det.detect(frame,tiled_ball=False)
for box in boxes:result['rknn_detections'].append({'class':int(box.cls[0]),'score':float(box.conf[0]),'bbox':box.xyxy[0].tolist()})
decoded=[]
for r in ref.detect(frame,tiled_ball=False):
    candidates=[d for d in boxes if int(d.cls[0])==int(r.cls[0])]
    def overlap(d):
        a,b=d.xyxy[0],r.xyxy[0];intersection=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
        return float(intersection/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection))
    best=max(candidates,key=overlap) if candidates else None
    decoded.append({'class':int(r.cls[0]),'iou':overlap(best) if best else 0,'confidence_absolute_error':abs(float(best.conf[0])-float(r.conf[0])) if best else 1})
result['decoded_detection_parity']=decoded
pose=Pose(ROOT/'models/pose_backbone.rknn','hybrid');pref=Pose(ROOT/'models/pose.onnx','onnx')
errors=[]
for d in boxes:
    if int(d.cls[0])==0:
        kp=np.array(pose.estimate(frame,d.xyxy[0]));kr=np.array(pref.estimate(frame,d.xyxy[0]));valid=kr[:,2]>.3
        if valid.any():errors.extend(np.linalg.norm(kp[valid,:2]-kr[valid,:2],axis=1).tolist())
result['pose_mean_keypoint_error_px']=float(np.mean(errors)) if errors else None
result['pose_max_keypoint_error_px']=float(np.max(errors)) if errors else None
result['pose_checked_keypoints']=len(errors)
(ROOT/'outputs').mkdir(exist_ok=True);(ROOT/'outputs/model_parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
det.close();ref.close();pose.close();pref.close()
assert min(result['detector_tensor_cosines'][::3])>.98
assert decoded and all(d['iou']>.95 and d['confidence_absolute_error']<.05 for d in decoded)
assert errors and np.mean(errors)<5
