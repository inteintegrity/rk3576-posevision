import sys,pathlib,cv2,numpy as np,json
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from basketball.pose import Pose
cap=cv2.VideoCapture(str(root/'upstream/input/sample2.mp4'));_,frame=cap.read();cap.release()
p=Pose(root/'models/pose.rknn');ref=Pose(root/'models/pose.onnx','onnx')
orig=p.rt.inference;data={}
def capture(*a,**k):
    out=orig(*a,**k);data['rgb']=k['inputs'][0];data['out']=out;return out
p.rt.inference=capture
box=[3499,837,3658,1190]
print('RKNN keypoints',p.estimate(frame,box));print('ONNX keypoints',ref.estimate(frame,box))
arr=(data['rgb'].astype(np.float32)-[123.675,116.28,103.53])/[58.395,57.12,57.375]
cpu=ref.rt.run(None,{ref.rt.get_inputs()[0].name:arr.transpose(0,3,1,2).astype(np.float32)})
for a,b in zip(data['out'],cpu):
    print('SHAPES',a.shape,b.shape,'RANGES',a.min(),a.max(),b.min(),b.max(),'COSINE',float(a.ravel()@b.ravel()/(np.linalg.norm(a)*np.linalg.norm(b))))
    print('RKNN argmax',a.argmax(-1),'CPU argmax',b.argmax(-1))
    print('RKNN first',a.ravel()[:20],'CPU first',b.ravel()[:20])
p.close();ref.close()
