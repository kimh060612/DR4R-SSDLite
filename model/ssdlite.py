from torch import nn
from model.backbone.mobilenetv2 import MobileNetV2
from model.backbone.mobilenext import MobileNeXt
from model.bbox.box_head import SSDBoxHead

def getBackboneModel(model_name, num_classes):
    if model_name == "MobileNetV2":
        return MobileNetV2(num_classes=num_classes)
    elif model_name == "MobileNeXt":
        return MobileNeXt(num_classes=num_classes)
    
class SSDLite(nn.Module):
    def __init__(self, cfg):
        super(SSDLite, self).__init__()
        self.backbone = getBackboneModel(cfg["model"]["backbone"], cfg["model"]["num_classes"])
        self.box_head = SSDBoxHead(cfg)
    
    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections