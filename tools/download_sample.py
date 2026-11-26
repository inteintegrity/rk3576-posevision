"""Download exact original Git blob with verified resumable byte ranges."""
import concurrent.futures, hashlib, pathlib, urllib.request, json, time, sys, subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
SIZE=95708286; SHA='6e47c96313a27f3d0e7268d402c69646fc92cb4b'
URL='https://api.github.com/repos/Yingbo-Jiao/PoseVision/git/blobs/'+SHA
if '--github-cli-auth' not in sys.argv:
    URL='https://raw.githubusercontent.com/Yingbo-Jiao/PoseVision/2d385598a26c187b2344b4f63b4ceee6ebdd2518/input/sample2.mp4'
if '--main' in sys.argv:URL='https://raw.githubusercontent.com/Yingbo-Jiao/PoseVision/main/input/sample2.mp4'
parts=ROOT/'.download-parts';parts.mkdir(exist_ok=True)
N=32;chunk=(SIZE+N-1)//N
headers={'Accept':'application/vnd.github.raw+json'}
if '--github-cli-auth' in sys.argv:
    # Use the existing CLI credential in memory only; never log or persist it.
    headers['Authorization']='Bearer '+subprocess.check_output(['gh','auth','token'],text=True).strip()
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
if '--github-cli-auth' in sys.argv:
    # Authenticated GitHub blob responses may ignore Range and return the full blob.
    dest=ROOT/'upstream/input/sample2.mp4';dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.download')
    with urllib.request.urlopen(urllib.request.Request(URL,headers=headers),timeout=300) as r, tmp.open('wb') as f:
        while block:=r.read(1024*1024):f.write(block);print(f.tell(),flush=True)
    data=tmp.read_bytes()
    assert len(data)==SIZE and hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==SHA
    tmp.replace(dest);print('VERIFIED original sample2.mp4',len(data),hashlib.sha256(data).hexdigest(),flush=True)
    sys.exit(0)
def fetch(i):
    start=i*chunk;end=min(SIZE,start+chunk)-1;p=parts/str(i)
    if p.exists() and p.stat().st_size==end-start+1:return
    if '--small-ranges' in sys.argv:
        def piece(s):
            e=min(end,s+131071)
            req=urllib.request.Request(URL+f'?range={s}',headers={**headers,'Range':f'bytes={s}-{e}'})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req,timeout=30) as r:
                        assert r.status==206 and r.headers.get('Content-Range')==f'bytes {s}-{e}/{SIZE}'
                        data=r.read()
                    assert len(data)==e-s+1
                    return data
                except Exception:
                    if attempt==4:raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:data=b''.join(ex.map(piece,range(start,end+1,131072)))
        assert len(data)==end-start+1
        p.write_bytes(data);print('part',i,len(data),flush=True);return
    req=urllib.request.Request(URL,headers={**headers,'Range':f'bytes={start}-{end}'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=240) as r:
                if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':raise ValueError('Unexpected range response')
                data=r.read()
            assert len(data)==end-start+1
            p.write_bytes(data);print('part',i,len(data),flush=True);return
        except Exception:
            if attempt==2:raise
            time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:list(ex.map(fetch,range(N)))
dest=ROOT/'upstream/input/sample2.mp4';dest.parent.mkdir(parents=True,exist_ok=True)
data=b''.join((parts/str(i)).read_bytes() for i in range(N))
assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==SHA,'Original Git blob hash mismatch'
dest.write_bytes(data);print('VERIFIED original sample2.mp4',len(data),hashlib.sha256(data).hexdigest(),flush=True)
