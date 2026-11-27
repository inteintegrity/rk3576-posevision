import time
import cv2
import numpy as np

EDGES=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
class Pose:
    def __init__(self,path,backend='rknn'):
        self.backend=backend;self.calls=0;self.seconds=0
        if backend in ('rknn','hybrid'):
            from rknnlite.api import RKNNLite
            self.rt=RKNNLite()
            if self.rt.load_rknn(str(path)) or self.rt.init_runtime():raise RuntimeError('Pose NPU initialization failed')
            if backend=='hybrid':
                import onnxruntime as ort
                opt=ort.SessionOptions();opt.intra_op_num_threads=2
                self.head=ort.InferenceSession(str(path.with_name('pose_head.onnx')),opt,providers=['CPUExecutionProvider'])
        self.head_seconds=0
        if backend=='onnx':
            import onnxruntime as ort
            opt=ort.SessionOptions();opt.intra_op_num_threads=2
            self.rt=ort.InferenceSession(str(path),opt,providers=['CPUExecutionProvider'])
    def estimate(self,frame,box):
        x1,y1,x2,y2=box;center=np.array([(x1+x2)/2,(y1+y2)/2]);w=max(x2-x1,2);h=max(y2-y1,2)
        w=max(w,h*192/256)*1.25;h=w*256/192
        src=np.float32([center-[w/2,h/2],center+[w/2,-h/2],center+[-w/2,h/2]])
        mat=cv2.getAffineTransform(src,np.float32([[0,0],[192,0],[0,256]]))
        rgb=cv2.cvtColor(cv2.warpAffine(frame,mat,(192,256)),cv2.COLOR_BGR2RGB)
        start=time.perf_counter()
        if self.backend in ('rknn','hybrid'):out=self.rt.inference(inputs=[rgb[None]],data_format=['nhwc'])
        else:
            arr=(rgb.astype(np.float32)-[123.675,116.28,103.53])/[58.395,57.12,57.375]
            out=self.rt.run(None,{self.rt.get_inputs()[0].name:arr.transpose(2,0,1)[None].astype(np.float32)})
        self.seconds+=time.perf_counter()-start;self.calls+=1
        if out is None:raise RuntimeError('Pose returned no tensors')
        if self.backend=='hybrid':
            start=time.perf_counter();out=self.head.run(None,{'features':np.asarray(out[0],dtype=np.float32)});self.head_seconds+=time.perf_counter()-start
        if not all(np.isfinite(o).all() for o in out) or all(np.count_nonzero(o)==0 for o in out):
            raise RuntimeError('Invalid pose tensors: all-zero/nonfinite output. Use verified hybrid or explicit ONNX CPU pose.')
        sx,sy=[np.asarray(o).reshape(17,-1) for o in out]
        pts=np.stack([sx.argmax(1)/2,sy.argmax(1)/2],1).astype(np.float32)
        pts=cv2.transform(pts[None],cv2.invertAffineTransform(mat))[0]
        # Same raw SimCC confidence convention as MMPose, without flip test.
        scores=np.minimum(sx.max(1),sy.max(1));return np.c_[pts,scores].tolist()
    def close(self):
        if self.backend in ('rknn','hybrid'):self.rt.release()
