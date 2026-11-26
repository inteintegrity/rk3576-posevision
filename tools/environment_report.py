import json, pathlib, platform, importlib.metadata, subprocess, sys
root=pathlib.Path(__file__).resolve().parents[1]
report={'machine':platform.machine(),'kernel':platform.release(),'python':platform.python_version()}
for name in ['numpy','opencv-python','onnxruntime','rknn-toolkit-lite2','deep-sort-realtime','fastapi','uvicorn']:
    try:report[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:report[name]=None
for name in ['fastapi','uvicorn']:
    try:report[name+'_imported_version']=__import__(name).__version__
    except ImportError:pass
for name,path in {'compatible':'/proc/device-tree/compatible','npu_frequency':'/sys/devices/platform/27700000.npu/devfreq/27700000.npu/cur_freq'}.items():
    try:report[name]=pathlib.Path(path).read_text().replace('\x00',',')
    except OSError:pass
print(json.dumps(report,indent=2))
if platform.machine()=='aarch64' and '--no-models' not in sys.argv:
    sys.path.insert(0,str(root))
    from basketball.detector import Detector
    from basketball.pose import Pose
    d=Detector(root/'models/detector.rknn');p=Pose(root/'models/pose_backbone.rknn','hybrid');d.close();p.close()
    print('YOLO NPU and hybrid RTMPose runtimes initialized successfully; run validate_models.py for numerical validation')
