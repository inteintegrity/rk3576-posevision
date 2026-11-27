import pathlib, torch, deep_sort_realtime, json, hashlib
from deep_sort_realtime.embedder.mobilenetv2_bottle import MobileNetV2_bottle
torch.set_num_threads(4)
m=MobileNetV2_bottle().eval()
p=pathlib.Path(deep_sort_realtime.__file__).parent/'embedder/weights/mobilenetv2_bottleneck_wts.pt'
m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))
torch.onnx.export(m,torch.zeros(1,3,224,224),str(pathlib.Path(__file__).resolve().parents[1]/'models/reid.onnx'),opset_version=12,dynamo=False,input_names=['image'],output_names=['features'],dynamic_axes={'image':{0:'batch'},'features':{0:'batch'}})
(pathlib.Path(__file__).resolve().parents[1]/'models/reid.json').write_text(json.dumps({'architecture':'DeepSORT ImageNet MobileNetV2 bottleneck','original_weights_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'onnx_input':['batch',3,224,224],'rknn_input':[1,224,224,3],'rknn_dtype':'uint8','color':'RGB','mean':[123.675,116.28,103.53],'std':[58.395,57.12,57.375],'output_features':1280,'trained':False},indent=2))
