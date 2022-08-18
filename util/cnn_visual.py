import functools
import operator
from functools import partial
from types import FunctionType

from captum import attr
from captum.attr import visualization as viz
import cv2
from imgaug import augmenters as iaa
import numpy as np
from scipy.ndimage import zoom
import torch
from torch.autograd import grad
import torch.nn.functional as F
from torch.utils.data import DataLoader


def get_visualizer(vis_method, model, net, **kwargs):
    if vis_method == 'IG':
        return CaptumVisualizer(vis_method, model, net, **kwargs)

class Visualizer(object):
    def __init__(self, model, net, **kwargs):
        self.model = model
        self.net = net
        self.opt = model.opt

    def cal_saliency_map(self, input, **kwargs):
        pass

    def init(self):
        pass

    def reset(self):
        pass

class CaptumVisualizer(Visualizer):
    
    def __init__(self, vis_method, model, net, **kwargs):
        super().__init__(model, net, **kwargs)
        self.vis_method = vis_method

    def _get_func(self):
        if self.vis_method == 'IG':
            return lambda func: attr.NoiseTunnel(attr.IntegratedGradients(func))

    def cal_saliency_map(self, input, **kwargs):
        worker = self._get_func()(partial(self.net.forward, indices=self.opt.vis_head))
        attributions = worker.attribute(input, target=self.model.instance_preds[self.opt.vis_head+1], nt_samples=8, nt_type='smoothgrad_sq')
        d = {'heatmap': []}
        for attribution in attributions:
            norm_attr = viz._normalize_image_attr(np.transpose(attribution.squeeze().cpu().detach().numpy(), (1,2,0)), 
                                                'positive', 
                                                outlier_perc=1) # (H, W)
            d['heatmap'].append(norm_attr)
        return d

