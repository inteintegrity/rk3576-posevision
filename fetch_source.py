import concurrent.futures, hashlib, json, pathlib, urllib.request
root=pathlib.Path(__file__).resolve().parent
revision=json.loads((root/'upstream_revision.json').read_text())['commit']
cache=root/'tree.json'
if not cache.exists():
    url='https://api.github.com/repos/Yingbo-Jiao/PoseVision/git/trees/'+revision+'?recursive=1'
    with urllib.request.urlopen(url,timeout=60) as response:cache.write_bytes(response.read())
tree=json.loads(cache.read_text())
assert tree['sha']==revision,'Unexpected upstream revision'
entries={x['path']:x for x in tree['tree'] if x['type']=='blob'}
paths=[x['path'] for x in tree['tree'] if x['type']=='blob' and (x['path'].endswith(('.py','.md','.txt','.yml')) and not x['path'].startswith('configs/') or x['path'] in ['yolo_detection/best.pt'] or x['path'].startswith('configs/body_2d_keypoint/rtmpose/body8/') or x['path'].startswith('configs/_base_/'))]
def fetch(path):
    dest=root/'upstream'/path; dest.parent.mkdir(parents=True,exist_ok=True)
    def valid(data):return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==entries[path]['sha']
    if dest.exists() and valid(dest.read_bytes()):return
    url='https://raw.githubusercontent.com/Yingbo-Jiao/PoseVision/'+tree['sha']+'/'+path
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url,timeout=180) as r: data=r.read()
            assert valid(data),'Upstream Git blob hash mismatch: '+path
            dest.write_bytes(data);print(path,len(data),flush=True);return
        except Exception:
            if attempt==3:raise
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(fetch,paths))
(root/'upstream_revision.json').write_text(json.dumps({'repository':'https://github.com/Yingbo-Jiao/PoseVision','commit':tree['sha']},indent=2))
