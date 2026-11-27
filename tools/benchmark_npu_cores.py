import pathlib,sys,cv2,numpy as np,time,json
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basketball.detector import letterbox
from basketball.runtime import configure_cpu
from rknnlite.api import RKNNLite
limits=configure_cpu();cap=cv2.VideoCapture(str(ROOT/'upstream/input/sample2.mp4'));ok,frame=cap.read();cap.release();assert ok
result={}
for model in ['detector','pose_backbone','reid']:
    rgb=letterbox(frame)[0] if model=='detector' else cv2.cvtColor(cv2.resize(frame[837:1190,3499:3658],(192,256) if model=='pose_backbone' else (224,224)),cv2.COLOR_BGR2RGB)
    reference=None;entries={}
    for mask in [0,3]:
        rt=RKNNLite();assert rt.load_rknn(str(ROOT/'models'/f'{model}.rknn'))==0
        code=rt.init_runtime(core_mask=mask)
        if code:entries[str(mask)]={'initialization_error':code};rt.release();continue
        times=[]
        for i in range(23):
            start=time.perf_counter();out=rt.inference(inputs=[rgb[None]],data_format=['nhwc']);elapsed=time.perf_counter()-start
            if i>=3:times.append(elapsed)
        if reference is None:reference=[a.copy() for a in out]
        entries[str(mask)]={'warmed_calls_per_second':len(times)/sum(times),'max_output_difference':max(float(np.abs(a-b).max()) for a,b in zip(out,reference))}
        rt.release()
    result[model]=entries
limits.restore_original_limits();(ROOT/'docs/npu-core-benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
