"""Check score-first DFL decoding against the committed full-grid reference."""
import pathlib,sys,subprocess,cv2,numpy as np,json,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from basketball.detector import Detector,letterbox,decode
source=subprocess.check_output(['git','show','d2e4326:basketball/detector.py'],cwd=ROOT,text=True)
namespace={'__file__':str(ROOT/'basketball/detector.py')};exec(compile(source,'baseline_decoder','exec'),namespace)
reference=namespace['decode'];d=Detector(ROOT/'models/detector.onnx','onnx');cap=cv2.VideoCapture(str(ROOT/'upstream/input/sample2.mp4'));result=[]
for index in [0,500,1000,1499]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();assert ok
    rgb,tr=letterbox(frame);outputs,_=d.run_raw(rgb)
    old=reference(outputs,tr,frame.shape,.12);new=decode(outputs,tr,frame.shape,.12)
    assert np.array_equal(old[2],new[2]) and np.array_equal(old[1],new[1])
    error=float(np.max(np.abs(old[0]-new[0]))) if len(new[0]) else 0
    assert error<.005
    result.append({'frame':index,'detections':len(new[0]),'max_coordinate_difference_original_pixels':error})
cap.release();(ROOT/'docs/decoder-parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
