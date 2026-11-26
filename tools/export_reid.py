import pathlib, torch, deep_sort_realtime
from deep_sort_realtime.embedder.mobilenetv2_bottle import MobileNetV2_bottle
torch.set_num_threads(4)
m=MobileNetV2_bottle().eval()
p=pathlib.Path(deep_sort_realtime.__file__).parent/'embedder/weights/mobilenetv2_bottleneck_wts.pt'
m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))
torch.onnx.export(m,torch.zeros(1,3,224,224),str(pathlib.Path(__file__).resolve().parents[1]/'models/reid.onnx'),opset_version=12,dynamo=False,input_names=['image'],output_names=['features'],dynamic_axes={'image':{0:'batch'},'features':{0:'batch'}})
