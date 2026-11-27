"""Compare final INT8 detections with the FP16 reference on held-out real frames."""
import pathlib,sys,cv2,json,time,collections,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basketball.detector import Detector
from basketball.runtime import configure_cpu
limits=configure_cpu();a=Detector(ROOT/'models/detector.rknn');b=Detector(ROOT/'models/detector_int8.rknn')
counts=collections.Counter();matches=collections.Counter();ious=[];extra=collections.Counter()
def iou(x,y):
    inter=np.maximum(np.minimum(x[2:],y[2:])-np.maximum(x[:2],y[:2]),0).prod()
    return float(inter/((x[2:]-x[:2]).prod()+(y[2:]-y[:2]).prod()-inter+1e-8))
cap=cv2.VideoCapture(str(ROOT/'upstream/input/sample2.mp4'))
indices=[37,139,271,413,559,701,853,997,1133,1291,1421]
for index in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();assert ok
    reference=a.detect(frame);actual=b.detect(frame);used=set()
    for det in reference:
        cls=int(det.cls[0]);counts[cls]+=1
        options=[(iou(det.xyxy[0],r.xyxy[0]),j) for j,r in enumerate(actual) if int(r.cls[0])==cls and j not in used]
        score,j=max(options,default=(0,-1))
        if score>=.5:matches[cls]+=1;ious.append(score);used.add(j)
    for j,det in enumerate(actual):
        if j not in used:extra[int(det.cls[0])]+=1
cap.release();a.close();b.close();limits.restore_original_limits()
retention={str(c):matches[c]/n for c,n in counts.items()}
result={'held_out_source_frames':indices,'fp16_detections':dict(counts),'matched_at_iou_0_5':dict(matches),'retention_against_fp16':retention,'extra_int8_detections':dict(extra),'mean_matched_iou':float(np.mean(ious)),'fp16_npu_calls_per_second':len(a.times)/sum(a.times),'int8_npu_calls_per_second':len(b.times)/sum(b.times)}
result['accepted']=retention.get('0',0)>=.95 and retention.get('1',0)>=.9 and retention.get('2',0)>=.95 and result['mean_matched_iou']>=.9 and extra[1]<=max(1,counts[1]*.2)
(ROOT/'docs/int8-evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
