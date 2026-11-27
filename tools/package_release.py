"""Package source, original assets, models and validation evidence; no caches."""
import hashlib,json,pathlib,zipfile,re
root=pathlib.Path(__file__).resolve().parents[1]
dest=root.parent/'rk3576-posevision-release.zip'
# Package only version-controlled final assets, respecting repository exclusions.
# Original footage/checkpoints remain obtainable through the acquisition scripts.
import subprocess
files=[root/name for name in subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines() if (root/name).is_file()]
manifest={str(p.relative_to(root)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files if p.name!='release-manifest.json'}
(root/'release-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in files:
        if p.name!='release-manifest.json':z.write(p,'rk3576-posevision/'+p.relative_to(root).as_posix())
    z.write(root/'release-manifest.json','rk3576-posevision/release-manifest.json')
print(dest,dest.stat().st_size)
