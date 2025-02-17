import numpy as np
import torch
import torch.nn as nn
from focal_loss import FocalLoss

def mixup(data, target1, target2, target3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target1 = target1[indices]
    shuffled_target2 = target2[indices]
    shuffled_target3 = target3[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    target1 = (target1, shuffled_target1, lam)
    target2 = (target2, shuffled_target2, lam)
    target3 = (target3, shuffled_target3, lam)

    return data, target1, target2, target3


def mixup_criterion(preds, targets):
    targets1, targets2, lam = targets

    targets1 = targets1.cuda()
    targets2 = targets2.cuda()

    criterion = nn.CrossEntropyLoss(reduction='mean')#nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(
        preds, targets2)


def mixup_criterion_with_focal_loss(preds, targets):
    targets1, targets2, lam = targets

    targets1 = targets1.cuda()
    targets2 = targets2.cuda()

    criterion = FocalLoss(168)#nn.CrossEntropyLoss(reduction='mean')#nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(
        preds, targets2)



def mixup_criterion_with_ohem(preds, targets):
    targets1, targets2, lam = targets

    targets1 = targets1.cuda()
    targets2 = targets2.cuda()

    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)#nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(
        preds, targets2)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, target1, target2, target3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target1 = target1[indices]
    shuffled_target2 = target2[indices]
    shuffled_target3 = target3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    target1 = (target1, shuffled_target1, lam)
    target2 = (target2, shuffled_target2, lam)
    target3 = (target3, shuffled_target3, lam)


    #targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
    return data, target1, target2, target3

def cutmix_criterion(preds, targets):
    targets1, targets2, lam = targets
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)






if __name__ == '__main__':
    img = torch.rand([4, 1, 128, 128])
    label = torch.tensor([[1,2,0],[2,1,0], [0,1,2], [2,0,1]])



