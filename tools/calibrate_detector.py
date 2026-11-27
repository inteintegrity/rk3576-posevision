"""Create calibration images from real full frames and the actual ball tile policy."""
import pathlib,sys,cv2,json,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basketball.detector import letterbox
indices=set(np.linspace(0,1499,32,dtype=int));records={}
for line in (ROOT/'outputs/verified-full/frames.jsonl').open():
    row=json.loads(line)
    if row['frame'] in indices:records[row['frame']]=row
dest=ROOT/'calibration';dest.mkdir(exist_ok=True);paths=[]
cap=cv2.VideoCapture(str(ROOT/'upstream/input/sample2.mp4'))
index=0
while True:
    ok,frame=cap.read()
    if not ok:break
    if index in indices:
        h,w=frame.shape[:2];size=min(h,w,1440)
        people=np.array([d['bbox'] for d in records[index]['detections'] if d['class_id'] in (0,2)])
        center=float(np.median((people[:,1]+people[:,3])/2)) if len(people) else h*.6
        y=int(np.clip(center-size/2,0,h-size));crops=[frame]+[frame[y:y+size,x:x+size] for x in np.linspace(0,w-size,int(np.ceil(w/size)),dtype=int)]
        for j,crop in enumerate(crops):
            rgb,_=letterbox(crop);path=dest/f'{index:04d}_{j}.png';cv2.imwrite(str(path),rgb[...,::-1]);paths.append(path.name)
    index+=1
cap.release();(dest/'dataset.txt').write_text('\n'.join(paths)+'\n');print('Calibration images:',len(paths))
