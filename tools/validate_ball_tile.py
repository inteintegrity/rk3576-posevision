import pathlib,sys,cv2,json
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from basketball.detector import Detector
cap=cv2.VideoCapture(str(root/'upstream/input/sample2.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,600);ok,frame=cap.read();cap.release();assert ok
d=Detector(root/'models/detector.rknn');out=d.detect(frame)
balls=[{'bbox':b.xyxy[0].tolist(),'score':float(b.conf[0])} for b in out if b.cls[0]==1]
result={'frame':600,'backend':'RK3576 NPU','inference_calls':len(d.times),'balls':balls}
print(json.dumps(result,indent=2));d.close();assert balls and max(b['score'] for b in balls)>.3
