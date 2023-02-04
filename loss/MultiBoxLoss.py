import torch.nn as nn
import torch.nn.functional as F
import torch

from loss.utils.boxloss import *

class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        num_classes = confidence.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = self.loss_fn(confidence.view(-1, num_classes), labels[mask])

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos