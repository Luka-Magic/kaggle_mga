import torch
from torch import nn


def neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x h x w)
        gt_regr (batch x h x w)
    '''
    pred = pred.unsqueeze(1).float()
    gt = gt.unsqueeze(1).float()

    pos_inds = gt.eq(1).float()  # peakがTrue
    neg_inds = gt.lt(1).float()  # peak以外がTrue
    neg_weights = torch.pow(1 - gt, 4)  # peak周りが0に近くなる

    loss = 0

    # peakだけ判定。pred->0.01~1=pos_loss->-4.5~0,
    pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, 3) * pos_inds
    # peak以外を判定。pred->0~0.99=neg_loss->0~-4.5, peak近くだと0~-2.0と弱まる
    neg_loss = torch.log(1 - pred + 1e-12) * \
        torch.pow(pred, 3) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class CenterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        pred = torch.sigmoid(pred[:, 0])
        loss = neg_loss(pred, gt)
        return loss


def neg_source_weight_loss(pred, gt, weight):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x h x w)
        gt_regr (batch x h x w)
        weight (batch)
    '''
    pred = pred.unsqueeze(1).float()
    gt = gt.unsqueeze(1).float()

    pos_inds = gt.eq(1).float()  # peakがTrue
    neg_inds = gt.lt(1).float()  # peak以外がTrue
    neg_weights = torch.pow(1 - gt, 4)  # peak周りが0に近くなる

    loss = 0

    # peakだけ判定。pred->0.01~1=pos_loss->-4.5~0,
    pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, 3) * pos_inds
    # peak以外を判定。pred->0~0.99=neg_loss->0~-4.5, peak近くだと0~-2.0と弱まる
    neg_loss = torch.log(1 - pred + 1e-12) * \
        torch.pow(pred, 3) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = (pos_loss * weight).sum()
    neg_loss = (neg_loss * weight).sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class CenterSourceWeightLoss(nn.Module):
    def __init__(self, weight_extracted=100.):
        super().__init__()
        self.weight_extracted = weight_extracted

    def forward(self, pred, gt, source):
        pred = torch.sigmoid(pred[:, 0])
        weight = self.weight_extracted * source + (1. - source)
        weight = weight.view(-1, 1, 1)
        loss = neg_loss(pred, gt, weight)
        return loss
