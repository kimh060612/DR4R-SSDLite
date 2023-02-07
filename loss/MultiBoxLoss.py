import torch.nn as nn
import torch.nn.functional as F
import torch

from loss.FocalLoss import FocalLoss
from loss.utils.boxloss import *

class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio, num_classes, use_focal=True):
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.num_classes = num_classes
        if use_focal:
            alpha = [[0.75]] * num_classes
            alpha[0] = [0.25]
            alpha = torch.Tensor(alpha)
            self.loss_fn = FocalLoss(alpha=alpha, gamma=2, class_num=num_classes, size_average=False)
        else: 
            self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = self.loss_fn(confidence.view(-1, self.num_classes), labels[mask])

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos