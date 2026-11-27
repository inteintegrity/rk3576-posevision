"""Compare ordered prefetch with sequential decoding and check encoder drain."""
import pathlib,sys,hashlib,cv2,imageio_ffmpeg,json
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basketball.video import VideoReader,VideoEncoder
source=ROOT/'upstream/input/sample2.mp4';cap=cv2.VideoCapture(str(source));hashes=[]
for _ in range(8):
    ok,frame=cap.read();assert ok;hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
cap.release();reader=VideoReader(source);dest=ROOT/'outputs/io-check';dest.mkdir(parents=True,exist_ok=True)
cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','960x540','-r','50','-i','-','-an','-c:v','libx264','-threads','2','-preset','ultrafast','-pix_fmt','yuv420p',str(dest/'result.mp4')]
writer=VideoEncoder(cmd,dest/'encoder.log')
try:
    for expected in hashes:
        ok,frame=reader.read();assert ok and hashlib.sha256(frame.tobytes()).hexdigest()==expected
        writer.write(cv2.resize(frame,(960,540)))
    writer.close()
finally:reader.release();writer.abort()
cap=cv2.VideoCapture(str(dest/'result.mp4'));count=0
while True:
    ok,frame=cap.read()
    if not ok:break
    assert frame.shape[:2]==(540,960);count+=1
fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert count==8 and fps==50
result={'ordered_original_frame_hashes_match':True,'encoded_and_decoded_frames':count,'fps':fps,'queue_capacity_frames':2}
(ROOT/'docs/video-io-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
