from torch import nn
import torch.nn.functional as F

from model.anchor.pbox import PriorBox
from model.utils.utils import *
from .inference import PostProcessor
from .box_predictor import SSDLiteBoxPredictor
from loss.FocalLoss import FocalLoss
from loss.MultiBoxLoss import MultiBoxLoss

class SSDBoxHead(nn.Module):
    def __init__(self, cfg):
        super(SSDBoxHead, self).__init__()
        self.cfg = cfg
        self.predictor = SSDLiteBoxPredictor(
            num_classes = cfg["model"]["num_classes"],
            loss_type = cfg["model"]["loss"],
            box_per_location=cfg["model"]["priors"]["box_per_location"],
            out_chans=cfg["model"]["out_channels"]
        )
        # 
        if cfg["model"]["loss"] == 'FocalLoss':
            self.loss_evaluator = FocalLoss(0.25, 2)
        else: # By default, we use MultiBoxLoss
            self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg["model"]["NEG_POS_RATIO"])

        self.post_processor = PostProcessor(
            IMG_SIZE = cfg["img_size"],
            loss_type = cfg["model"]["loss"],
            confidence_thr = cfg["test"]["confidence_thr"],
            max_num = cfg["test"]["max_num"]
        )
        self.priors = None

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred, gt_labels, gt_boxes)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg["img_size"], self.cfg["model"]["priors"])().to(bbox_pred.device)
        if self.cfg["model"]["loss"] == 'FocalLoss':
            scores = cls_logits.sigmoid()
        else:
            scores = F.softmax(cls_logits, dim=2)

        boxes = convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg["model"]["CENTER_VARIANCE"], self.cfg["model"]["SIZE_VARIANCE"]
        )
        
        boxes = center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        return detections, {}