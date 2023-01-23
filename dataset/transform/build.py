from .transform import *
from .transform_target import SSDTargetTransform
from model.anchor.pbox import PriorBox

def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(cfg["PIXEL_MEAN"]),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg["img_size"]),
            SubtractMeans(cfg["PIXEL_MEAN"]),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg["img_size"]),
            SubtractMeans(cfg["PIXEL_MEAN"]),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform

def build_target_transform(cfg):
    transform = SSDTargetTransform(
        PriorBox(
            img_size=cfg["img_size"], 
            prior_config=cfg["model"]["priors"]
        )(),
        cfg["model"]["CENTER_VARIANCE"],
        cfg["model"]["SIZE_VARIANCE"],
        cfg["model"]["THRESHOLD"]
    )
    return transform
