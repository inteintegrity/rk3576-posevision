import json, pathlib, threading, uuid, shutil
from concurrent.futures import ThreadPoolExecutor
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from basketball.pipeline import process
from basketball.court import validate_calibration
ROOT=pathlib.Path(__file__).resolve().parent;OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
app=FastAPI(title='Basketball Vision • RK3576');pool=ThreadPoolExecutor(max_workers=1);lock=threading.Lock();jobs={}
app.mount('/web',StaticFiles(directory=ROOT/'web'),name='web')

def folder(job):
    if not __import__('re').fullmatch('[a-f0-9]{32}',job):raise HTTPException(404)
    p=OUT/job
    if not p.is_dir():raise HTTPException(404)
    return p
def status(job,**values):
    jobs.setdefault(job,{}).update(values)
    p=folder(job);tmp=p/'status.tmp';tmp.write_text(json.dumps(jobs[job]));tmp.replace(p/'status.json')
@app.get('/')
def index():return FileResponse(ROOT/'web/index.html')
@app.post('/api/videos')
async def upload(file:UploadFile=File(...)):
    if not file.filename.lower().endswith('.mp4'):raise HTTPException(400,'Upload an MP4 file')
    job=uuid.uuid4().hex;p=OUT/job;p.mkdir();size=0
    try:
        with (p/'input.mp4').open('wb') as f:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>512*1024**2:raise HTTPException(413,'Maximum upload size is 512 MB')
                f.write(chunk)
        inspect_video(p);status(job,state='ready',frames=0);return {'id':job}
    except Exception:
        (p/'input.mp4').unlink(missing_ok=True);raise
    finally:await file.close()
def inspect_video(p):
    cap=cv2.VideoCapture(str(p/'input.mp4'))
    try:
        ok,frame=cap.read()
        if not ok:raise HTTPException(400,'MP4 cannot be decoded')
        cv2.imwrite(str(p/'first.jpg'),frame)
    finally:cap.release()
@app.post('/api/sample')
def sample():
    src=ROOT/'upstream/input/sample2.mp4'
    if not src.exists():raise HTTPException(404,'Original sample2.mp4 has not been downloaded')
    job=uuid.uuid4().hex;p=OUT/job;p.mkdir();shutil.copyfile(src,p/'input.mp4');inspect_video(p);status(job,state='ready',frames=0);return {'id':job}
@app.post('/api/jobs/{job}/run')
def run(job:str,options:dict):
    p=folder(job)
    if jobs.get(job,{}).get('state') in ['running','queued','complete']:raise HTTPException(409,'Create a new job to process again')
    try:cal=validate_calibration(options.get('calibration'))
    except (ValueError,KeyError,TypeError) as e:raise HTTPException(400,str(e))
    backend=options.get('pose','hybrid')
    if backend not in ['hybrid','onnx','off']:raise HTTPException(400,'Unknown pose backend')
    if not lock.acquire(False):raise HTTPException(409,'NPU is busy processing another video')
    status(job,state='queued')
    def worker():
        try:
            status(job,state='running');report=process(p/'input.mp4',p,cal,backend,update=lambda s:status(job,**s));status(job,state='complete',**report)
        except Exception as e:
            import traceback
            (p/'error.log').write_text(traceback.format_exc());status(job,state='failed',error=str(e))
        finally:lock.release()
    pool.submit(worker);return {'id':job}
@app.get('/api/jobs/{job}')
def get_status(job:str):
    p=folder(job);s=jobs.get(job) or json.loads((p/'status.json').read_text())
    if s.get('state') in ['running','queued'] and job not in jobs:s['state']='interrupted'
    return dict(s)
@app.get('/api/jobs/{job}/{name}')
def get_file(job:str,name:str):
    if name not in ['input.mp4','first.jpg','preview.jpg','result.mp4','report.json','frames.jsonl']:raise HTTPException(404)
    p=folder(job)/name
    if not p.exists():raise HTTPException(404)
    if name in ['result.mp4','report.json','frames.jsonl'] and get_status(job).get('state')!='complete':raise HTTPException(409,'Processing is not complete')
    return FileResponse(p,media_type='video/mp4' if name.endswith('.mp4') else None)
