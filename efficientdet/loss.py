import torch
import torch.nn as nn
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import postprocess, invert_affine, display


def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


def bbox_ciou(box1, box2, loss='ciou'):
    """
    Returns the CIoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)

    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1, 1).expand(N, M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1, -1).expand(N, M)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    outer_rect_x1 = torch.min(b1_x1.unsqueeze(1), b2_x1)
    outer_rect_y1 = torch.min(b1_y1.unsqueeze(1), b2_y1)
    outer_rect_x2 = torch.max(b1_x2.unsqueeze(1), b2_x2)
    outer_rect_y2 = torch.max(b1_y2.unsqueeze(1), b2_y2)
    outer_rect_diag = (outer_rect_x2 - outer_rect_x1) ** 2 + (outer_rect_y2 - outer_rect_y1) ** 2

    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1
    b1_xc, b1_yc = b1_x1 + b1_w / 2, b1_y1 + b1_h / 2
    b2_xc, b2_yc = b2_x1 + b2_w/ 2, b2_y1 + b2_h / 2
    centers_dist = (b1_xc.unsqueeze(1) - b2_xc) ** 2 + (b1_yc.unsqueeze(1) - b2_yc) ** 2
    u = centers_dist / outer_rect_diag

    b1_atan = torch.atan(b1_w / b1_h).view(-1, 1).expand(N, M)
    b2_atan = torch.atan(b2_w / b2_h).view(1, -1).expand(N, M)
    arctan = b2_atan - b1_atan
    v = torch.pow(arctan / (np.pi / 2), 2)
    alpha = v / (1 - iou + v + 1e-8)
    ciou = iou - u - alpha * v
    ciou = torch.clamp(ciou, min=-1, max=1.)

    return iou, ciou


class FocalLoss(nn.Module):
    def __init__(self, embedding_size=0, n_ids=0):
        super(FocalLoss, self).__init__()
        self.id_classifier = nn.Linear(embedding_size, n_ids) if n_ids > 0 else None
        self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, classifications, regressions, embeddings, anchors, annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        embeddings_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            embedding = embeddings[j, :, :]

            bbox_annotation = annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    embeddings_losses.append(torch.tensor(0).to(dtype).cuda())
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    classification_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    embeddings_losses.append(torch.tensor(0).to(dtype))
                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(torch.tensor(0).to(dtype))

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if num_positive_anchors > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                # Embeddings
                if self.id_classifier is not None:
                    assigned_embedding = embedding[positive_indices]
                    id_targets = assigned_annotations[:, -1].long()
                    id_logits = self.id_classifier(assigned_embedding)
                    emb_loss = self.id_loss(id_logits, id_targets)
                    embeddings_losses.append(emb_loss.sum() / num_positive_anchors.to(dtype))

                # Regression
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                # Smoothed L1
                # gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                # gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                # gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                # gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights
                #
                # # efficientdet style
                # gt_widths = torch.clamp(gt_widths, min=1)
                # gt_heights = torch.clamp(gt_heights, min=1)
                #
                # targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                # targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                # targets_dw = torch.log(gt_widths / anchor_widths_pi)
                # targets_dh = torch.log(gt_heights / anchor_heights_pi)
                #
                # targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                # targets = targets.t()
                #
                # regression_diff = torch.abs(targets - regression[positive_indices, :])
                #
                # regression_loss = torch.where(
                #     torch.le(regression_diff, 1.0 / 9.0),
                #     0.5 * 9.0 * torch.pow(regression_diff, 2),
                #     regression_diff - 0.5 / 9.0
                # )
                # regression_losses.append(regression_loss.mean())

                # Regression CIoU
                assigned_regression = regression[positive_indices, :]
                w = assigned_regression[:, 3].exp() * anchor_widths_pi
                h = assigned_regression[:, 2].exp() * anchor_heights_pi

                y_centers = assigned_regression[:, 0] * anchor_heights_pi + anchor_ctr_y_pi
                x_centers = assigned_regression[:, 1] * anchor_widths_pi + anchor_ctr_x_pi

                ymin, ymax = y_centers - h / 2., y_centers + h / 2.
                xmin, xmax = x_centers - w / 2., x_centers + w / 2.
                assigned_regression = torch.stack([xmin, ymin, xmax, ymax], dim=1)
                iou, ciou = bbox_ciou(assigned_regression, assigned_annotations[:, :4])
                ciou_loss = 1. - torch.mean(torch.diag(ciou))
                regression_losses.append(ciou_loss)

            else:
                if torch.cuda.is_available():
                    embeddings_losses.append(torch.tensor(0).to(dtype).cuda())
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    embeddings_losses.append(torch.tensor(0).to(dtype))
                    regression_losses.append(torch.tensor(0).to(dtype))

        # debug
        imgs = kwargs.get('imgs', None)
        if imgs is not None:
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            obj_list = kwargs.get('obj_list', None)
            out = postprocess(imgs.detach(),
                              torch.stack([anchors[0]] * imgs.shape[0], 0).detach(),
                              regressions.detach(), classifications.detach(), embeddings.detach(),
                              regressBoxes, clipBoxes,
                              0.5, 0.3)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            display(out, imgs, obj_list, imshow=False, imwrite=True)

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True), \
               torch.stack(embeddings_losses).mean(dim=0, keepdim=True)
