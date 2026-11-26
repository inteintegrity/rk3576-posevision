import argparse, pathlib, json, time, hashlib
from rknn.api import RKNN
p=argparse.ArgumentParser();p.add_argument('model',choices=['detector','pose','pose_backbone']);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];src=root/'models'/f'{a.model}.onnx';dest=src.with_suffix('.rknn')
r=RKNN(verbose=True);start=time.perf_counter()
try:
    mean,std=([[0,0,0]],[[255,255,255]]) if a.model=='detector' else ([[123.675,116.28,103.53]],[[58.395,57.12,57.375]])
    assert r.config(target_platform='rk3576',mean_values=mean,std_values=std,optimization_level=3)==0
    assert r.load_onnx(model=str(src))==0
    assert r.build(do_quantization=False)==0
    assert r.export_rknn(str(dest))==0
    dest.with_suffix('.build.json').write_text(json.dumps({'target':'rk3576','precision':'fp16','seconds':time.perf_counter()-start,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'rknn_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))
finally:r.release()
