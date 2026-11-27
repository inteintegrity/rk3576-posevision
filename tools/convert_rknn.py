import argparse, pathlib, json, time, hashlib
from rknn.api import RKNN
p=argparse.ArgumentParser();p.add_argument('model',choices=['detector','pose','pose_backbone','reid']);p.add_argument('--dataset');a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];src=root/'models'/f'{a.model}.onnx';dest=src.with_name(src.stem+('_int8' if a.dataset else '')+'.rknn')
r=RKNN(verbose=True);start=time.perf_counter()
try:
    mean,std=([[0,0,0]],[[255,255,255]]) if a.model=='detector' else ([[123.675,116.28,103.53]],[[58.395,57.12,57.375]])
    assert r.config(target_platform='rk3576',mean_values=mean,std_values=std,optimization_level=3)==0
    shape={'inputs':['image'],'input_size_list':[[1,3,224,224]]} if a.model=='reid' else {}
    assert r.load_onnx(model=str(src),**shape)==0
    assert r.build(do_quantization=bool(a.dataset),dataset=a.dataset)==0
    assert r.export_rknn(str(dest))==0
    dest.with_suffix('.build.json').write_text(json.dumps({'target':'rk3576','precision':'int8' if a.dataset else 'fp16','seconds':time.perf_counter()-start,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'rknn_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))
finally:r.release()
