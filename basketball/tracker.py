import cv2,time,os
from pathlib import Path
import numpy as np
import onnxruntime as ort
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self,model,backend=None):
        self.backend=backend or os.environ.get('BASKETBALL_REID','rknn')
        self.npu_seconds=0.;self.npu_calls=0
        if self.backend=='rknn':
            from rknnlite.api import RKNNLite
            self.rt=RKNNLite()
            self.core_mask=int(os.environ.get('BASKETBALL_REID_CORES','3'))
            if self.rt.load_rknn(str(Path(model).with_suffix('.rknn'))) or self.rt.init_runtime(core_mask=self.core_mask):raise RuntimeError('ReID NPU initialization failed; set BASKETBALL_REID=onnx for explicit CPU ReID')
        elif self.backend=='onnx':
            opt=ort.SessionOptions();opt.intra_op_num_threads=2
            self.rt=ort.InferenceSession(str(model),opt,providers=['CPUExecutionProvider'])
        else:raise ValueError('ReID backend must be rknn or onnx')
        # Independent association prevents player/ball/referee class switches.
        self.trackers={c:DeepSort(max_age=30,n_init=2 if c==1 else 3,embedder=None,max_cosine_distance=.3,nn_budget=100) for c in (0,1,2)}
        self.mean=np.array([.485,.456,.406],np.float32)
        self.std=np.array([.229,.224,.225],np.float32)
        self.reid_seconds=0.;self.association_seconds=0.
    def update(self,frame,detections):
        start=time.perf_counter();result=[];crops=[];groups={c:[] for c in self.trackers}
        for d in detections:
            cls=int(d.cls[0])
            if cls not in groups:continue
            x1,y1,x2,y2=d.xyxy[0].astype(int);crop=frame[y1:y2,x1:x2]
            if crop.size==0:continue
            index=len(crops);crops.append(cv2.resize(crop[...,::-1],(224,224)))
            det=([float(x1),float(y1),float(x2-x1),float(y2-y1)],float(d.conf[0]),cls)
            groups[cls].append((index,det))
        features=None
        if crops:
            if self.backend=='rknn':
                values=[]
                for crop in crops:
                    tick=time.perf_counter();outputs=self.rt.inference(inputs=[crop[None]],data_format=['nhwc'])
                    self.npu_seconds+=time.perf_counter()-tick;self.npu_calls+=1
                    if outputs is None:raise RuntimeError('ReID NPU returned no tensors')
                    values.append(outputs[0].reshape(-1))
                features=np.asarray(values,np.float32)
                if not np.isfinite(features).all() or np.any(np.linalg.norm(features,axis=1)<1e-8):raise RuntimeError('Invalid ReID features')
            else:
                arr=np.asarray(crops,np.float32);arr*=np.float32(1/255)
                arr-=self.mean;arr/=self.std
                features=self.rt.run(None,{'image':np.ascontiguousarray(arr.transpose(0,3,1,2))})[0]
        self.reid_seconds+=time.perf_counter()-start
        start=time.perf_counter()
        for cls,tracker in self.trackers.items():
            dets=[d for _,d in groups[cls]]
            feats=features[[i for i,_ in groups[cls]]] if groups[cls] else []
            for t in tracker.update_tracks(dets,embeds=feats):
                # Do not render Kalman-only predictions as current detections.
                if t.is_confirmed() and t.time_since_update==0:
                    result.append({'id':f'{cls}-{t.track_id}','class_id':cls,'bbox':t.to_ltrb(orig=True).tolist(),'confidence':float(t.get_det_conf() or 0)})
        self.association_seconds+=time.perf_counter()-start
        return result
    def close(self):
        if self.backend=='rknn':self.rt.release()

class Teams:
    def __init__(self):self.centers=None;self.samples=[];self.votes={}
    def classify(self,frame,tracks):
        features=[]
        for t in tracks:
            if t['class_id']!=0:continue
            x1,y1,x2,y2=map(int,t['bbox']);h,w=frame.shape[:2]
            # Central upper torso avoids hardwood and spectators near bbox edges.
            a,b=max(0,int(y1+(y2-y1)*.18)),min(h,int(y1+(y2-y1)*.5))
            c,d=max(0,int(x1+(x2-x1)*.25)),min(w,int(x1+(x2-x1)*.75))
            crop=frame[a:b,c:d];t['team']=-1
            if crop.size:
                hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
                feature=np.median(hsv.reshape(-1,3),axis=0)[1:].astype(np.float32)
                features.append((t,feature))
        if self.centers is None:
            self.samples.extend(f for _,f in features)
            if len(self.samples)>=20:
                cv2.setRNGSeed(0)
                _,_,centers=cv2.kmeans(np.array(self.samples,np.float32),2,None,(cv2.TERM_CRITERIA_MAX_ITER,30,.1),5,cv2.KMEANS_PP_CENTERS)
                # Require observable appearance separation before assigning teams.
                if np.linalg.norm(centers[0]-centers[1])>=25:self.centers=centers[np.argsort(centers[:,0])]
                self.samples=self.samples[-200:]
        if self.centers is not None:
            for t,f in features:
                label=int(np.linalg.norm(self.centers-f,axis=1).argmin())
                votes=self.votes.setdefault(t['id'],[0,0]);votes[label]+=1;t['team']=int(np.argmax(votes))
