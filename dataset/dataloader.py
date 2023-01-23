from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from model.bbox.bbox import BBox
from dataset.coco import COCODataset
from dataset.transform.build import build_target_transform, build_transforms
from config.confg import getDatasetConfig

class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.is_train:
            list_targets = transposed_batch[1]
            targets = BBox(
                {
                    key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]
                }
            )
        else:
            targets = None
        return images, targets, img_ids

def build_train_dataloader(cfg):
    train_transform = build_transforms(cfg, is_train=True)
    target_transform = build_target_transform(cfg)
    args = getDatasetConfig(is_train=True)
    dataset = COCODataset(
        args["data_dir"], 
        args["annotation_file"], 
        transform=train_transform, 
        target_transform=target_transform, 
        remove_empty=True
    )
    return DataLoader(
        dataset=dataset, 
        batch_size=cfg["train"]["bsz"], 
        shuffle=True, 
        num_workers=cfg["train"]["num_workers"],
        collate_fn=BatchCollator(True)
    )

def build_validation_dataloader(cfg):
    val_transform = build_transforms(cfg, is_train=True)
    args = getDatasetConfig(is_train=False)
    dataset = COCODataset(
        args["data_dir"], 
        args["annotation_file"], 
        transform=val_transform, 
        target_transform=None
    )
    return DataLoader(
        dataset=dataset,
        batch_size=cfg["validation"]["bsz"],
        num_workers=cfg["validation"]["num_workers"],
        collate_fn=BatchCollator(False)
    )