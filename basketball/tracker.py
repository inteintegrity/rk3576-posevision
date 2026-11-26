import cv2
import numpy as np
import onnxruntime as ort
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self,model):
        opt=ort.SessionOptions();opt.intra_op_num_threads=2
        self.rt=ort.InferenceSession(str(model),opt,providers=['CPUExecutionProvider'])
        # Independent association prevents player/ball/referee class switches.
        self.trackers={c:DeepSort(max_age=30,n_init=2 if c==1 else 3,embedder=None,max_cosine_distance=.3) for c in (0,1,2)}
    def update(self,frame,detections):
        result=[]
        for cls,tracker in self.trackers.items():
            dets=[];crops=[]
            for d in detections:
                if int(d.cls[0])!=cls:continue
                x1,y1,x2,y2=d.xyxy[0].astype(int);crop=frame[y1:y2,x1:x2]
                if crop.size==0:continue
                crops.append(cv2.resize(crop[...,::-1],(224,224)))
                dets.append(([float(x1),float(y1),float(x2-x1),float(y2-y1)],float(d.conf[0]),cls))
            feats=[]
            if crops:
                arr=np.asarray(crops,np.float32)/255
                arr=(arr-[.485,.456,.406])/[.229,.224,.225]
                feats=self.rt.run(None,{'image':arr.transpose(0,3,1,2).astype(np.float32)})[0]
            for t in tracker.update_tracks(dets,embeds=feats):
                # Do not render Kalman-only predictions as current detections.
                if t.is_confirmed() and t.time_since_update==0:
                    result.append({'id':f'{cls}-{t.track_id}','class_id':cls,'bbox':t.to_ltrb(orig=True).tolist(),'confidence':float(t.get_det_conf() or 0)})
        return result

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
