import torch.nn as nn
from torch.nn.modules import loss


def get_cls_criterion(loss_type, opt):
    if opt.class_num > 1:
        if loss_type == 'ce':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError("not support criterion named: {}".format(loss_type))
    else:
        if loss_type == 'ce':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError("not support criterion named: {}".format(loss_type))

def get_reg_criterion(loss_type, opt):
    if loss_type.lower() == 'l1':
        return nn.L1Loss()
    elif loss_type.lower() in ('l2', 'mse'):
        return nn.MSELoss()