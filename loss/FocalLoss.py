import torch.nn as nn
import torch.nn.functional as F
import torch
from loss.utils.boxloss import *
from loss.SmoothL1Loss import SmoothL1Loss, convert_to_one_hot

class FocalSigmoidLossFuncV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, label, alpha, gamma):
        logits = logits.float()
        coeff = torch.empty_like(logits).fill_(1 - alpha)
        coeff[label == 1] = alpha

        probs = torch.sigmoid(logits)
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        probs_gamma = probs ** gamma
        probs_1_gamma = (1. - probs) ** gamma

        ctx.coeff = coeff
        ctx.probs = probs
        ctx.log_probs = log_probs
        ctx.log_1_probs = log_1_probs
        ctx.probs_gamma = probs_gamma
        ctx.probs_1_gamma = probs_1_gamma
        ctx.label = label
        ctx.gamma = gamma

        term1 = probs_1_gamma * log_probs
        term2 = probs_gamma * log_1_probs
        loss = torch.where(label == 1, term1, term2).mul_(coeff).neg_()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        coeff = ctx.coeff
        probs = ctx.probs
        log_probs = ctx.log_probs
        log_1_probs = ctx.log_1_probs
        probs_gamma = ctx.probs_gamma
        probs_1_gamma = ctx.probs_1_gamma
        label = ctx.label
        gamma = ctx.gamma

        term1 = (1. - probs - gamma * probs * log_probs).mul_(probs_1_gamma).neg_()
        term2 = (probs - gamma * (1. - probs) * log_1_probs).mul_(probs_gamma)

        grads = torch.where(label == 1, term1, term2).mul_(coeff).mul_(grad_output)
        return grads, None, None, None


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth_l1_loss = SmoothL1Loss()

    # def forward(self, logits, label):
    def forward(self, confidence, predicted_locations, labels, gt_locations):
        num_classes = confidence.size(-1)
        confidence = confidence.view(-1, confidence.size(-1))
        labels = labels.view(-1)

        pos_mask = labels > 0
        num_pos = pos_mask.data.long().sum()

        labels = convert_to_one_hot(labels, num_classes + 1)
        loss = FocalSigmoidLossFuncV2.apply(confidence, labels, self.alpha, self.gamma)

        predicted_locations = predicted_locations.view(-1, 4)[pos_mask]
        gt_locations = gt_locations.view(-1, 4)[pos_mask]
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        
        return smooth_l1_loss.sum() / num_pos, 5 * loss.sum() / num_pos

class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        # compute loss
        confidence = confidence.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(confidence).fill_(1 - self.alpha)
            alpha[labels == 1] = self.alpha

        probs = torch.sigmoid(confidence)
        pt = torch.where(labels == 1, probs, 1 - probs)
        ce_loss = self.crit(confidence, labels.double())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss