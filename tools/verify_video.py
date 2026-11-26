"""Decode every processed frame and verify source timing and output dimensions."""
import argparse,cv2,json,pathlib,hashlib
p=argparse.ArgumentParser();p.add_argument('video');a=p.parse_args();root=pathlib.Path(__file__).resolve().parents[1]
cap=cv2.VideoCapture(a.video);assert cap.isOpened()
fps=cap.get(cv2.CAP_PROP_FPS);fourcc=int(cap.get(cv2.CAP_PROP_FOURCC));codec=''.join(chr((fourcc>>(8*i))&255) for i in range(4));count=0;shape=None
while True:
    ok,frame=cap.read()
    if not ok:break
    shape=frame.shape[:2];assert shape==(540,1360)
    if count==600:cv2.imwrite(str(root/'docs/result-preview.jpg'),frame)
    count+=1
cap.release();assert count==1500 and abs(fps-50)<.01 and codec.lower() in ['avc1','h264']
result={'decoded_frames':count,'fps':fps,'duration_seconds':count/fps,'width':shape[1],'height':shape[0],'codec_fourcc':codec,'bytes':pathlib.Path(a.video).stat().st_size,'sha256':hashlib.sha256(pathlib.Path(a.video).read_bytes()).hexdigest()}
(root/'docs/video-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
