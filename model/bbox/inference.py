import torch
import torch.nn as nn
from .bbox import BBox
from model.utils.utils import batched_nms

class PostProcessor:
    def __init__(self, IMG_SIZE, loss_type, confidence_thr, max_num):
        super().__init__()
        self.width = IMG_SIZE
        self.height = IMG_SIZE
        self.cls_thresh = 0.5
        self.cls_loss = loss_type
        self.confidence_threshold = confidence_thr
        self.max_num = max_num

    def __call__(self, detections):
        batches_scores, batches_boxes = detections
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        for batch_id in range(batch_size):
            scores, boxes = batches_scores[batch_id], batches_boxes[batch_id]  # (N, #CLS) (N, 4)
            num_boxes = scores.shape[0]
            num_classes = scores.shape[1]

            boxes = boxes.view(num_boxes, 1, 4).expand(num_boxes, num_classes, 4)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, num_classes).expand_as(scores)

            # remove predictions with the background label
            if self.cls_loss != 'FocalLoss':
                boxes = boxes[:, 1:]
                scores = scores[:, 1:]
                labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            indices = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

            boxes[:, 0::2] *= self.width
            boxes[:, 1::2] *= self.height

            keep = batched_nms(boxes, scores, labels, self.confidence_threshold)
            # keep only topk scoring predictions
            keep = keep[:self.max_num]

            # if self.cls_loss == 'FocalLoss':
            #     boxes, scores, labels = boxes[keep], scores[keep], labels[keep] + 1
            # else:
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            container = BBox(boxes=boxes, labels=labels, scores=scores)
            container.img_width = self.width
            container.img_height = self.height
            results.append(container)
            
        return results
