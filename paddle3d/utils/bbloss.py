import numpy as np
import paddle
def limit( ang):
    ang = ang % (2 * np.pi)

    ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

    ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

    return ang


def ang_weight(pred, gt):

    a2 = paddle.abs(paddle.sin(pred - gt)) 

    return 1-a2

def compute_iou(x,w,y,l):
    zmax1 = x + w * 0.5
    zmin1 = x - w * 0.5
    zmax2 = y + l * 0.5
    zmin2 = y - l * 0.5
    z_overlap = paddle.clip(paddle.minimum(zmax1, zmax2) - paddle.maximum(zmin1, zmin2), min=0)
    
    all_lap = paddle.clip(paddle.maximum(zmax1, zmax2) - paddle.minimum(zmin1, zmin2),min=0)
    iou = z_overlap / all_lap
    return iou

def bb_loss(pred, target):
    iouw = compute_iou(pred[..., 0], pred[..., 3], target[..., 0], target[..., 3])
    ioul = compute_iou(pred[..., 1], pred[..., 4], target[..., 1], target[..., 4])
    iouh = compute_iou(pred[..., 2], pred[..., 5], target[..., 2], target[..., 5])

    a_p = limit(pred[..., 6])
    a_g = limit(target[..., 6])
    ioua = ang_weight(a_p, a_g)

    iou = iouw*ioul*iouh*ioua

    diff_angle = pred[:, -1] - target[:, -1]
    angle_factor = 1.25 * (1.0 - paddle.abs(paddle.cos(diff_angle)))

    center_dist_square = paddle.pow(target[:, 0:3] - pred[:, 0:3], 2).sum(-1)

    finall_loss = 1-iou + angle_factor + center_dist_square

    return finall_loss*1.5
    #may have question
class APLoss(paddle.nn.Layer):
    def __init__(self):
        super(APLoss, self).__init__()

    def forward(self, logits, targets):
        classification_grads, classification_losses = AP_loss(logits, targets)
        return classification_losses


def AP_loss(logits, targets):
    delta = 1.0

    grad = paddle.zeros(logits.shape)
    metric = paddle.zeros(1)

    if paddle.max(targets) <= 0:
        return grad, metric

    labels_p = (targets == 1)
    fg_logits = logits[labels_p]
    threshold_logit = paddle.min(fg_logits) - delta #-0.9

    ######## Ignore those negative j that satisfy (L_{ij}=0 for all positive i), to accelerate the AP-loss computation.
    valid_labels_n = ((targets == 0) & (logits >= threshold_logit))
    valid_bg_logits = logits[valid_labels_n]
    valid_bg_grad = paddle.zeros(len(valid_bg_logits))
    ########

    fg_num = len(fg_logits)
    prec = paddle.zeros(fg_num)
    order = paddle.argsort(fg_logits)
    max_prec = 0

    for ii in order:
        tmp1 = fg_logits - fg_logits[ii]
        tmp1 =paddle.clip(tmp1 / (2 * delta) + 0.5, min=0, max=1)
        tmp2 = valid_bg_logits - fg_logits[ii]
        tmp2 = paddle.clip(tmp2 / (2 * delta) + 0.5, min=0, max=1)
        a = paddle.sum(tmp1) + 0.5
        b = paddle.sum(tmp2)
        tmp2 /= (a + b)
        current_prec = a / (a + b)
        if (max_prec <= current_prec):
            max_prec = current_prec
        else:
            tmp2 *= ((1 - max_prec) / (1 - current_prec))
        valid_bg_grad += tmp2
        prec[ii] = max_prec

    grad[valid_labels_n] = valid_bg_grad
    grad[labels_p] = -(1 - prec)

    fg_num = max(fg_num, 1)

    grad /= (fg_num)

    metric = paddle.sum(prec, dim=0, keepdim=True) / fg_num

    return grad, 1 - metric