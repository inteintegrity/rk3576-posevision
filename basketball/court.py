import cv2
import numpy as np

def validate_calibration(data):
    if data is None:return None
    src=np.asarray(data['image_points'],np.float32);dst=np.asarray(data['court_points'],np.float32)
    if src.shape!=(4,2) or dst.shape!=(4,2) or not np.isfinite(src).all() or not np.isfinite(dst).all():raise ValueError('Provide four finite corresponding point pairs')
    if np.any(src<0) or np.any(src>1) or np.any(dst<0) or np.any(dst>1):raise ValueError('Points must be normalized to [0,1]')
    for points in (src,dst):
        if not cv2.isContourConvex(points) or abs(cv2.contourArea(points))<.001:raise ValueError('Select four distinct convex points in matching perimeter order')
    return {'image_points':src.tolist(),'court_points':dst.tolist()}

class Court:
    def __init__(self,data):
        self.data=validate_calibration(data);self.H=None
        self.previous=None;self.mapping_status='uncalibrated';self.inliers=0
        if data:self.H=cv2.getPerspectiveTransform(np.float32(data['image_points']),np.float32(data['court_points']))
        if data:self.mapping_status='calibrated'
    def update_camera(self,frame,detections):
        if self.H is None or self.mapping_status=='lost':return
        gray=cv2.cvtColor(cv2.resize(frame,(960,540)),cv2.COLOR_BGR2GRAY)
        if self.previous is not None:
            P=np.diag([960.,540.,1.]);mask=np.zeros_like(gray)
            try:
                outline=cv2.perspectiveTransform(np.float32([[[0,0],[1,0],[1,1],[0,1]]]),np.linalg.inv(self.H))[0]*[960,540]
                outline=np.clip(outline,-5000,5000).astype(np.int32)
                cv2.fillConvexPoly(mask,outline,255)
                # Suppress moving people and ball in the motion-estimation mask.
                h,w=frame.shape[:2]
                for d in detections:
                    x1,y1,x2,y2=d.xyxy[0]*[960/w,540/h,960/w,540/h]
                    cv2.rectangle(mask,(int(x1)-12,int(y1)-12),(int(x2)+12,int(y2)+12),0,-1)
                pts=cv2.goodFeaturesToTrack(self.previous,maxCorners=250,qualityLevel=.01,minDistance=8,mask=mask)
                if pts is None or len(pts)<12:raise ValueError('Insufficient floor features')
                nxt,st,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,pts,None,winSize=(21,21),maxLevel=3)
                back,bst,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,nxt,None,winSize=(21,21),maxLevel=3)
                good=(st[:,0]>0)&(bst[:,0]>0)&(np.linalg.norm(pts[:,0]-back[:,0],axis=1)<1.5)
                a,b=pts[good],nxt[good]
                if len(a)<12:raise ValueError('Insufficient floor matches')
                motion,inliers=cv2.findHomography(a,b,cv2.RANSAC,2.0)
                self.inliers=int(inliers.sum()) if inliers is not None else 0
                if motion is None or self.inliers<12 or self.inliers/len(a)<.6:raise ValueError('Camera registration lost')
                if not np.isfinite(motion).all() or not .8<abs(np.linalg.det(motion[:2,:2]))<1.25:raise ValueError('Camera cut or unstable registration')
                self.H=self.H@np.linalg.inv(P)@np.linalg.inv(motion)@P
                self.H/=self.H[2,2];self.mapping_status='camera-tracked'
            except (ValueError,np.linalg.LinAlgError,cv2.error):
                # Never continue painting stale calibrated locations after lost registration.
                self.mapping_status='lost'
        self.previous=gray
    def project(self,tracks,shape):
        h,w=shape[:2]
        for t in tracks:
            t['court_position']=None
            if self.H is None or self.mapping_status=='lost':continue
            x1,y1,x2,y2=t['bbox'];p=np.float32([(x1+x2)/2/w,(y1+y2)/2/h if t['class_id']==1 else y2/h])
            q=cv2.perspectiveTransform(p[None,None],self.H)[0,0]
            if np.isfinite(q).all() and (q>=0).all() and (q<=1).all():t['court_position']=q.tolist()
    def draw(self,tracks):
        image=np.full((540,400,3),(30,42,38),np.uint8)
        cv2.rectangle(image,(22,45),(378,515),(180,190,180),2)
        cv2.line(image,(22,280),(378,280),(180,190,180),1)
        cv2.circle(image,(200,280),45,(180,190,180),1)
        for y in [45,400]:cv2.rectangle(image,(130,y),(270,y+115),(180,190,180),1)
        cv2.putText(image,'TACTICAL COURT',(30,27),0,.7,(240,240,240),1)
        for t in tracks:
            q=t.get('court_position')
            if q is not None:
                x,y=int(22+q[0]*356),int(45+q[1]*470)
                color=(30,180,255) if t['class_id']==1 else ((230,130,50) if t.get('team')==0 else (80,210,100))
                cv2.circle(image,(x,y),5,color,-1);cv2.putText(image,t['id'],(x+6,y),0,.35,(255,255,255),1)
        if self.H is None:cv2.putText(image,'Calibration required',(50,320),0,.65,(190,190,190),1)
        elif self.mapping_status=='lost':cv2.putText(image,'Camera tracking lost',(40,320),0,.65,(190,190,190),1)
        else:cv2.putText(image,self.mapping_status,(30,535),0,.45,(190,190,190),1)
        return image
