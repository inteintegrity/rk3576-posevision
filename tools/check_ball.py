"""Inspect real ball scores on full frames and overlapping original-pixel tiles."""
import pathlib,sys,cv2,numpy as np,json
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from basketball.detector import Detector,letterbox,decode
d=Detector(root/'models/detector.onnx','onnx');cap=cv2.VideoCapture(str(root/'upstream/input/sample2.mp4'))
for index in [300,600,900,1200]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();assert ok
    result=[];h,w=frame.shape[:2]
    for x,y in [(-1,-1)]+[(x,y) for y in [0,h-1440] for x in [0,1200,w-1440]]:
        crop=frame if x<0 else frame[y:y+1440,x:x+1440]
        rgb,tr=letterbox(crop);out,_=d.run_raw(rgb)
        b,s,c=decode(out,tr,crop.shape,confidence=.08)
        result.append({'tile':[x,y],'max_ball_score':max(float(o[:,1].max()) for o in out[1::3]),'balls':[{'box':bb.tolist(),'score':float(ss)} for bb,ss,cc in zip(b,s,c) if cc==1]})
    print(json.dumps({'frame':index,'scores':result}),flush=True)
cap.release()
