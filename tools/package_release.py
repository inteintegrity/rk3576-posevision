"""Package source, original assets, models and validation evidence; no caches."""
import hashlib,json,pathlib,zipfile,re
root=pathlib.Path(__file__).resolve().parents[1]
dest=root.parent/'rk3576-posevision-release.zip'
excluded={'.git','.deps','.venv','.convert','.download-parts','__pycache__','outputs'}
files=[]
for p in root.rglob('*'):
    rel=p.relative_to(root)
    if not p.is_file() or any(c in excluded for c in rel.parts):continue
    if p.suffix in ['.log','.gz','.download'] or p.name in ['sample-api.mp4','rtmpose.zip']:continue
    if re.fullmatch(r'check\d+.*\.onnx',p.name):continue
    files.append(p)
manifest={str(p.relative_to(root)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files if p.name!='release-manifest.json'}
(root/'release-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in files:
        if p.name!='release-manifest.json':z.write(p,'rk3576-posevision/'+p.relative_to(root).as_posix())
    z.write(root/'release-manifest.json','rk3576-posevision/release-manifest.json')
print(dest,dest.stat().st_size)
