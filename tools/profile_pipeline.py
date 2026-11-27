"""Profile genuine full-pipeline processing on an explicitly selected source window."""
import argparse, collections, cProfile, json, pathlib, pstats, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basketball import pipeline
p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=500);p.add_argument('--frames',type=int,default=100);p.add_argument('--output',required=True);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True);stages=collections.defaultdict(float)
def wrap(cls,name,label):
    original=getattr(cls,name)
    def measured(*args,**kwargs):
        start=time.perf_counter()
        try:return original(*args,**kwargs)
        finally:stages[label]+=time.perf_counter()-start
    setattr(cls,name,measured)
for cls,name,label in [(pipeline.Detector,'detect','detection'),(pipeline.Tracker,'update','tracking_reid'),(pipeline.Teams,'classify','teams'),(pipeline.Pose,'estimate','pose'),(pipeline.Court,'update_camera','camera'),(pipeline.Court,'project','projection'),(pipeline.Court,'draw','court_drawing')]:wrap(cls,name,label)
original_cap=pipeline.cv2.VideoCapture
class Capture:
    def __init__(self,*args,**kwargs):
        self.cap=original_cap(*args,**kwargs)
        if a.start:self.cap.set(pipeline.cv2.CAP_PROP_POS_FRAMES,a.start)
    def __getattr__(self,name):return getattr(self.cap,name)
    def read(self):
        start=time.perf_counter()
        try:return self.cap.read()
        finally:stages['decode']+=time.perf_counter()-start
pipeline.cv2.VideoCapture=Capture
profile=cProfile.Profile();profile.enable()
report=pipeline.process(ROOT/'upstream/input/sample2.mp4',out,json.loads((ROOT/'docs/sample2-calibration.json').read_text()) if a.start==0 else None,max_frames=a.frames)
profile.disable();report['source_start_frame']=a.start;report['stages_seconds']=dict(stages)
(out/'profile.json').write_text(json.dumps(report,indent=2))
with (out/'profile.txt').open('w') as f:pstats.Stats(profile,stream=f).strip_dirs().sort_stats('cumtime').print_stats(45)
print(json.dumps(report,indent=2))
