"""Verify completed WebUI report, HTTP video seeking and exported traces."""
import argparse,json,pathlib,requests
p=argparse.ArgumentParser();p.add_argument('--base',default='http://192.168.3.120:8093');p.add_argument('--job',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];s=requests.Session();s.trust_env=False;url=f'{a.base}/api/jobs/{a.job}'
state=s.get(url,timeout=30);state.raise_for_status();state=state.json();assert state['state']=='complete',state
r=s.get(url+'/report.json',timeout=30);r.raise_for_status();report=r.json();assert report['frames']==1500 and not report['partial_run']
r=s.get(url+'/result.mp4',headers={'Range':'bytes=0-127'},timeout=30);assert r.status_code==206 and len(r.content)==128 and 'ftyp' in r.content.decode('latin1')
trace=s.get(url+'/frames.jsonl',timeout=60);trace.raise_for_status();rows=[json.loads(line) for line in trace.text.splitlines()];assert len(rows)==1500
assert any(d['class_id']==1 for f in rows for d in f['detections'])
assert any(t.get('keypoints') for f in rows for t in f['tracks'])
assert any(t.get('court_position') for f in rows for t in f['tracks'])
assert all(len(t['keypoints'])==17 for f in rows for t in f['tracks'] if 'keypoints' in t)
result={'job_id':a.job,'state':'complete','verified':['1500 source frames processed','real ball detections present','17-keypoint pose observations present','calibrated court positions present','MP4 byte-range seeking returns HTTP 206','JSONL contains 1500 frame records'],'report':report}
(root/'docs/completed-web-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
