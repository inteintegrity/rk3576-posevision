"""Export the exact PoseVision RTMPose-s checkpoint; no training or custom ops."""
import sys, types, pathlib, json, hashlib
import torch, onnx
# MMCV compiled ops are not used by CSPNeXt / RTMCCHead. Fail if one is invoked.
ext=types.ModuleType('mmcv._ext')
def unavailable(*a,**k): raise RuntimeError('Compiled MMCV op is not allowed in this exporter')
ext.__file__='export-only-unavailable-extension'
ext.__getattr__=lambda name: unavailable
sys.modules['mmcv._ext']=ext
from mmengine.config import Config
from mmpose.registry import MODELS
from mmpose.utils import register_all_modules
register_all_modules()
root=pathlib.Path(__file__).resolve().parents[1]
cfg=Config.fromfile(str(root/'upstream/configs/body_2d_keypoint/rtmpose/body8/rtmpose-s_8xb256-420e_body8-256x192.py'))
cfg.model.backbone.init_cfg=None
model=MODELS.build(cfg.model).cpu().eval()
state=torch.load(root/'rtmpose.pth',map_location='cpu',weights_only=False)['state_dict']
model.load_state_dict(state,strict=True)
class Export(torch.nn.Module):
    def __init__(self,m):super().__init__();self.model=m
    def forward(self,x):return self.model.head(self.model.backbone(x))
torch.set_num_threads(4)
dest=root/'models/pose.onnx'
torch.onnx.export(Export(model).eval(),torch.zeros(1,3,256,192),str(dest),opset_version=14,input_names=['image'],output_names=['simcc_x','simcc_y'])
class Backbone(torch.nn.Module):
    def __init__(self,m):super().__init__();self.backbone=m.backbone
    def forward(self,x):return self.backbone(x)[0]
class Head(torch.nn.Module):
    def __init__(self,m):super().__init__();self.head=m.head
    def forward(self,x):return self.head((x,))
torch.onnx.export(Backbone(model).eval(),torch.zeros(1,3,256,192),str(root/'models/pose_backbone.onnx'),opset_version=14,input_names=['image'],output_names=['features'])
torch.onnx.export(Head(model).eval(),torch.zeros(1,512,8,6),str(root/'models/pose_head.onnx'),opset_version=14,input_names=['features'],output_names=['simcc_x','simcc_y'])
onnx.checker.check_model(onnx.load(str(dest)))
(root/'models/pose.json').write_text(json.dumps({'checkpoint':'rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth','sha256':hashlib.sha256((root/'rtmpose.pth').read_bytes()).hexdigest(),'input':[1,3,256,192],'outputs':[[1,17,384],[1,17,512]],'mean':[123.675,116.28,103.53],'std':[58.395,57.12,57.375],'rgb':True,'split_ratio':2,'flip_test':False},indent=2))
