import os

def getDatasetConfig(is_train=True):
    root = '/media/hdd2/cocodataset'
    if is_train:
        return {
            "data_dir": os.path.join(root, 'train2017/'),
            "annotation_file": os.path.join(root, 'annotations/instances_train2017.json'),
        }
    else :
        return {
            "data_dir": os.path.join(root, 'val2017/'),
            "annotation_file": os.path.join(root, 'annotations/instances_val2017.json'),
        }

def getConfig(backbone="MobileNetV2"):
    model_cfg = [ 96, 1280, 512, 256, 256, 64 ] if backbone == "MobileNetV2" else [ 576, 1280, 512, 256, 256, 128 ]
    return {
        "model": {
            "backbone": backbone,
            "loss": "FocalLoss", # "MultiBoxLoss",
            "num_classes": 81,
            "CENTER_VARIANCE": 0.1,
            "SIZE_VARIANCE": 0.2,
            "NEG_POS_RATIO": 3,
            "THRESHOLD": 0.5,
            "out_channels": model_cfg,
            "priors": {
                "feature_map": [
                    20, 10, 5, 3, 2, 1
                ],
                "stride": [
                    16, 32, 64, 120, 160, 320
                ],
                "min_size": [
                    45, 90, 135, 180, 225, 270
                ],
                "max_size": [
                    90, 135, 180, 225, 270, 315
                ],
                "aspect_ratio": [
                    [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]
                ],
                "box_per_location": [
                    6, 6, 6, 6, 6, 4
                ],
                "clip": True,
            }
        },
        "img_size": 320,
        "PIXEL_MEAN": [123, 117, 104],
        "train": {
            "epoch": 200,
            "bsz": 48,
            "lr": 2e-3,
            "gamma": 0.1,
            "weight_decay": 5e-4,
            "lr_scheduler": "CosineAnnealingWarmRestarts",
            "num_workers": 12,
            "log_step": 100
        },
        "validation": {
            "bsz": 48,
            "num_workers": 12
        },
        "test": {
            "nms_thr": 0.5,
            "confidence_thr": 0.5,
            "max_num": 100
        },
        "checkpoint_dir": './checkpoint'
    }