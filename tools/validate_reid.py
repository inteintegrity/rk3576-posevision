"""Validate NPU appearance vectors against the unchanged CPU ONNX model."""
import pathlib,sys,json,cv2,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basketball.tracker import Tracker
from basketball.runtime import configure_cpu
limits=configure_cpu();cpu=Tracker(ROOT/'models/reid.onnx','onnx');npu=Tracker(ROOT/'models/reid.onnx','rknn')
from basketball.detector import Detector
cap=cv2.VideoCapture(str(ROOT/'upstream/input/sample2.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,500);ok,frame=cap.read();cap.release();assert ok
detector=Detector(ROOT/'models/detector.rknn');detections=detector.detect(frame);detector.close()
cosines=[];errors=[]
for d in detections:
    if int(d.cls[0]) not in (0,1,2):continue
    x1,y1,x2,y2=map(int,d.xyxy[0]);crop=frame[y1:y2,x1:x2]
    if not crop.size:continue
    rgb=cv2.resize(crop[...,::-1],(224,224));arr=rgb.astype(np.float32)/255;arr-=cpu.mean;arr/=cpu.std
    ref=cpu.rt.run(None,{'image':np.ascontiguousarray(arr.transpose(2,0,1)[None])})[0].ravel()
    actual=npu.rt.inference(inputs=[rgb[None]],data_format=['nhwc'])[0].ravel()
    cosines.append(float(actual@ref/(np.linalg.norm(actual)*np.linalg.norm(ref))))
    errors.append(float(np.max(np.abs(actual/np.linalg.norm(actual)-ref/np.linalg.norm(ref)))))
result={'source_frame':500,'checked_crops':len(cosines),'feature_cosines':cosines,'minimum_cosine':min(cosines),'maximum_normalized_component_error':max(errors)}
(ROOT/'docs/reid-parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
npu.close();cpu.close();limits.restore_original_limits();assert cosines and min(cosines)>.999
