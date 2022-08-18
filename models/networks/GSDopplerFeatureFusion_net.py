import torch
import torch.nn as nn
import torch.nn.functional as F
from . import BACKBONES, init_weights

class GSDopplerFeatureFusionNet(nn.Module):

    def __init__(self, backbones, archs, in_channels, heads, share_weight=False, **kwargs):
        super(GSDopplerFeatureFusionNet, self).__init__()

        assert len(backbones) == len(archs)
        self.share_weight = share_weight
        self.feature_dims = []

        if self.share_weight:
            self.backbone1, feature_dim = BACKBONES[backbones[0]](arch=archs[0], in_channel=in_channels[0], **kwargs)
            self.feature_dims = [feature_dim for i in range(len(backbones))]
        else:
            for i in range(len(backbones)):
                backbone, feature_dim = BACKBONES[backbones[i]](arch=archs[i], in_channel=in_channels[i], **kwargs)
                setattr(self, 'backbone{}'.format(i+1), backbone)
                self.feature_dims.append(feature_dim)
        self.heads = heads
        self.build_heads(**kwargs)

    def build_heads(self, **kwargs):
        for i, head_config in enumerate(self.heads):
            feature_dim = 0
            for backbone_idx in head_config[0]:
                feature_dim += self.feature_dims[backbone_idx]
            fc1 = nn.Linear(feature_dim, 256)
            fc2 = nn.Linear(256, head_config[1])
            fc1.apply(init_weights)
            fc2.apply(init_weights)
            fc = nn.Sequential(fc1, fc2)
            setattr(self, 'head{}'.format(i+1), fc)

    def _forward_heads(self, features, indices=None):
        y_list = []
        if indices is None:
            for i, (backbones, _) in enumerate(self.heads):
                total_features = [features[backbone_idx] for backbone_idx in backbones]
                total_feature = torch.cat(total_features, dim=1)
                fc = getattr(self, 'head{}'.format(i+1))
                y = fc(total_feature)
                y_list.append(y)
            return y_list
        elif isinstance(indices, list):
            for i in indices:
                backbones, _ = self.heads[i]
                total_features = [features[backbone_idx] for backbone_idx in backbones]
                total_feature = torch.cat(total_features, dim=1)
                fc = getattr(self, 'head{}'.format(i+1))
                y = fc(total_feature)
                y_list.append(y)
            return y_list
        elif isinstance(indices, int):
            backbones, _ = self.heads[indices]
            total_features = [features[backbone_idx] for backbone_idx in backbones]
            total_feature = torch.cat(total_features, dim=1)
            fc = getattr(self, 'head{}'.format(indices+1))
            y = fc(total_feature)
            return y

    def forward(self, *input, indices=None):
        assert len(input) == len(self.feature_dims)
        features = []
        if self.share_weight:
            for i in range(len(input)):
                y = self.backbone1(input[i])
                if isinstance(y, tuple):
                    y = y[-1]
                if isinstance(y, list):
                    y = torch.cat(y, dim=1)
                y = y.view(y.size(0), -1)
                features.append(y)
        else:
            for i in range(len(input)):
                y = getattr(self, 'backbone{}'.format(i+1))(input[i])
                if isinstance(y, tuple):
                    y = y[-1]
                if isinstance(y, list):
                    y = torch.cat(y, dim=1)
                y = y.view(y.size(0), -1)
                features.append(y)

        y_list = self._forward_heads(features, indices)
        return y_list