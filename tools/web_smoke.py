"""Integration test against the real board; starts a full original-video job."""
import requests,json,pathlib
root=pathlib.Path(__file__).resolve().parents[1];s=requests.Session();s.trust_env=False;base='http://192.168.3.120:8093'
assert s.get(base).status_code==200
r=s.post(base+'/api/sample');r.raise_for_status();sample=r.json()['id']
assert s.get(f'{base}/api/jobs/{sample}/first.jpg').status_code==200
bad=s.post(f'{base}/api/jobs/{sample}/run',json={'calibration':{'image_points':[[0,0]]*4,'court_points':[[0,0]]*4}});assert bad.status_code==400
with (root/'upstream/input/sample2.mp4').open('rb') as f:r=s.post(base+'/api/videos',files={'file':('sample2.mp4',f,'video/mp4')},timeout=180)
r.raise_for_status();job=r.json()['id']
cal=json.loads((root/'docs/sample2-calibration.json').read_text())
r=s.post(f'{base}/api/jobs/{job}/run',json={'pose':'hybrid','calibration':cal});r.raise_for_status()
assert s.post(f'{base}/api/jobs/{sample}/run',json={'pose':'off'}).status_code==409
assert s.get(f'{base}/api/jobs/{job}/result.mp4').status_code in [404,409]
result={'job_id':job,'url':f'{base}/api/jobs/{job}','tested':['GET English UI','original sample selection','first-frame decode','degenerate calibration rejection','MP4 upload','hybrid NPU job submission','concurrent NPU job rejection','incomplete MP4 withheld']}
(root/'docs/web-integration.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
