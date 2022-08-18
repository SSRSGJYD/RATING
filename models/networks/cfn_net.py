import torch
import torch.nn as nn
from . import BACKBONES, init_weights

class CFNNet(nn.Module):
    '''
    Context-free Network for self-supervised learning by solving jigsaw puzzles.
    '''
    def __init__(self, backbones, archs, in_channels, heads, num_crops=9, **kwargs):
        super(CFNNet, self).__init__()

        assert len(backbones) == len(archs)
        self.backbone1, feature_dim = BACKBONES[backbones[0]](arch=archs[0], in_channel=in_channels[0], **kwargs)
        self.heads = heads
        self.num_crops = num_crops
        self.fc = nn.Linear(feature_dim * self.num_crops, self.heads[0][1])
        self.fc.apply(init_weights)

    def forward(self, *input, **kwargs):
        y = self.backbone1(input[0])
        if isinstance(y, tuple):
            y = y[-1]
        y = y.view(y.size(0), -1).contiguous() # (N, C)
        y = y.view(y.size(0) // self.num_crops, -1) # (N/n, n*C)
        y = self.fc(y)
        return [y]