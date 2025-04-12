import numpy as np
import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1).to(pred.device)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1).long(), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def focal_loss(pred, gold, alpha=1, gamma=2, reduction='mean'):
    '''
    Calculate Focal Loss for multi-class segmentation.

    Arguments:
    pred -- predicted logits for each class, shape (batch_size, num_points, num_classes)
    gold -- ground truth labels, shape (batch_size, num_points)
    alpha -- balancing factor, default is 1
    gamma -- focusing parameter, default is 2
    reduction -- specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'

    Returns:
    focal_loss -- calculated Focal loss
    '''

    batch_size, num_points, num_classes = pred.shape

    # Ensure the tensors are on the same device and gold is of integer type
    gold = gold.to(pred.device).long()

    # Apply softmax to get probabilities
    pred_prob = F.softmax(pred, dim=-1)  # Shape (batch_size, num_points, num_classes)

    # One-hot encode the ground truth labels
    gold_one_hot = F.one_hot(gold, num_classes=num_classes).float()  # Shape (batch_size, num_points, num_classes)

    # Compute the cross-entropy loss
    ce_loss = -gold_one_hot * torch.log(pred_prob + 1e-9)  # Shape (batch_size, num_points, num_classes)

    # Compute the modulating factor
    pt = torch.sum(gold_one_hot * pred_prob, dim=-1)  # Shape (batch_size, num_points)
    modulating_factor = (1 - pt) ** gamma

    # Compute the focal loss
    focal_loss = alpha * modulating_factor.unsqueeze(-1) * ce_loss

    if reduction == 'mean':
        focal_loss = torch.mean(focal_loss)
    elif reduction == 'sum':
        focal_loss = torch.sum(focal_loss)

    return focal_loss