import json
import os
from typing import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from .base_model import BaseModel
import config
from .model_option import model_dict
from optim import get_cls_criterion
from util import mkdir, get_visualizer

class ClsModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        parser = BaseModel.modify_commandline_options(parser)

        # model
        parser.add_argument('--backbones', type=str, default='resnet')
        parser.add_argument('--archs', type=str, default='18')
        parser.add_argument('--share_weight', type=int, default=0, help='share backbone in GSDopplerFeatureFusion net')
        parser.add_argument('--in_channels', type=str, default='3', help='input channels of each backbone')
        parser.add_argument('--heads', type=str, default='0_2', help='head of GSDopplerFeatureFusion net')
        parser.add_argument('--frozen_stages', type=int, default=0)
        parser.add_argument('--output_layers', type=int, default=1)
        parser.add_argument('--bottleneck_dim', type=int, default=256, help='bottleneck dim')
        parser.add_argument('--fc_mult', type=float, default=1)
        parser.add_argument('--recall_thresholds', type=str, default=None)

        # optimization
        parser.add_argument('--imagenet_pretrain', type=int, default=1)
        parser.add_argument('--loss_type', type=str, default='ce')
        parser.add_argument('--loss_weights', type=str, default='0,1')

        parser.set_defaults(valid_metric='instance_auc', scheduler_metric='instance_auc')
        return parser

    def __init__(self, opt):
        super().__init__(opt)

        self.opt.class_num = len(self.opt.classes)
        self.net_names = ['cls']
        model = model_dict[self.opt.method_name]['name']
        param_dict = dict()
        for param_name in model_dict[self.opt.method_name]['params']:
            param_dict[param_name] = getattr(self.opt, param_name)
        self.net_cls = model(pretrained=(opt.l_state=='train' and opt.imagenet_pretrain), **param_dict)

        self.buffer_g_instance_scores = [[] for i in range(1+len(self.opt.heads))]
        self.buffer_g_instance_preds = [[] for i in range(1+len(self.opt.heads))]
        self.buffer_g_instance_labels = [[] for i in range(1+len(self.opt.heads))]
        self.buffer_g_ids = []

        if 'jigsaw' in self.opt.name or 'heads=0_4' in self.opt.name or 'heads=0,1_4' in self.opt.name:
            self.opt.valid_metric = 'instance_accuracy'
            self.opt.scheduler_metric = 'instance_accuracy'

        # visualization
        self.visualizers = []
        if self.opt.visualize and self.opt.l_state != 'train':
            for vis_method in self.opt.vis_methods:
                visualizer = get_visualizer(vis_method, self, self.net_cls)
                self.visualizers.append(visualizer)
            name = opt.name
            attr_list = ['recall_thresholds', 'vis_head', 'vis_methods']
            for attr_name in attr_list:
                attr = getattr(self.opt, attr_name)
                name += ',{}={}'.format(attr_name, attr)
            self.vis_dir = os.path.join(opt.vis_dir, name+opt.remark)
            mkdir(self.vis_dir)
        
        # load pretrained models
        if self.opt.l_state == 'train' and self.opt.method_name == 'GSDopplerFeatureFusion':
            if self.opt.task == 'GS':
                if self.opt.classes_ == '3,012-3,012':
                    pretrain_classes = '01,23-01,23'
                elif self.opt.classes_ == '1,023-1,023':
                    pretrain_classes = '0,123-0,123'
                elif self.opt.classes_ == '2,013-2,013':
                    pretrain_classes = '3,012-3,012'
                else:
                    pretrain_classes = None

                if pretrain_classes is not None:
                    base_dir = config.GS_checkpoints
                    dirs = os.listdir(base_dir)
                    name = list(filter(lambda x: 'archs={}'.format(self.opt.archs_) in x and 'frozen_stages={}'.format(self.opt.frozen_stages) in x and 'classes={}'.format(pretrain_classes) in x and 'split={}'.format(self.opt.split) in x and 'seed={}'.format(self.opt.seed) in x, dirs))[0]
                    state_dict = torch.load(os.path.join(base_dir, name, 'optimal_net_cls.pth'), map_location='cpu')
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        if 'backbone' in k:
                            new_state_dict[k] = v
                    self.net_cls.load_state_dict(new_state_dict, strict=False)
                    print(name, 'GS checkpoint loaded')
                else:
                    # load self supervised pretrained models
                    base_dir = config.GS_checkpoints
                    dirs = os.listdir(base_dir)
                    name = list(filter(lambda x: 'archs={}'.format(self.opt.archs_) in x and 'GS_jigsaw' in x and 'split={}'.format(self.opt.split) in x, dirs))[0]
                    state_dict = torch.load(os.path.join(base_dir, name, 'optimal_net_cls.pth'), map_location='cpu')
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        if 'backbone' in k:
                            new_state_dict[k] = v
                    self.net_cls.load_state_dict(new_state_dict, strict=False)
                    print(name, 'GS checkpoint loaded')

            elif self.opt.task == 'DOPPLER':
                if self.opt.label == 'GS':
                    # load pretrained GS model
                    base_dir = config.GS_checkpoints
                    dirs = os.listdir(base_dir)
                    name = list(filter(lambda x: 'label={}'.format(self.opt.label) in x and 'archs={}'.format(self.opt.archs_) in x and 'frozen_stages={}'.format(self.opt.frozen_stages) in x and 'classes={}'.format(self.opt.classes_) in x and 'split={}'.format(self.opt.split) in x and 'seed={}'.format(self.opt.seed) in x, dirs))[0]
                    state_dict = torch.load(os.path.join(base_dir, name, 'optimal_net_cls.pth'), map_location='cpu')
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        if 'backbone' in k:
                            new_state_dict[k] = v
                    self.net_cls.load_state_dict(new_state_dict, strict=False)
                    print(name, 'GS checkpoint loaded')

            elif self.opt.task == 'GSDOPPLER':
                # load pretrained GS models
                base_dir = config.GS_checkpoints
                dirs = os.listdir(base_dir)
                name = list(filter(lambda x: 'label={}'.format(self.opt.label) in x and 'archs=18' in x and 'classes={}'.format(self.opt.classes_) in x and 'split={}'.format(self.opt.split) in x and 'seed={}'.format(self.opt.seed) in x, dirs))[0]
                state_dict = torch.load(os.path.join(base_dir, name, 'optimal_net_cls.pth'), map_location='cpu')
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'backbone' in k:
                        new_state_dict[k] = v
                self.net_cls.load_state_dict(new_state_dict, strict=False)
                print(name, 'GS checkpoint loaded')

                # load pretrained DOPPLER models
                base_dir = config.DOPPLER_checkpoints
                dirs = os.listdir(base_dir)
                name = list(filter(lambda x: 'label={}'.format(self.opt.label) in x and 'archs=18' in x and 'classes={}'.format(self.opt.classes_) in x and 'split={}'.format(self.opt.split) in x and 'seed={}'.format(self.opt.seed) in x, dirs))[0]
                state_dict = torch.load(os.path.join(base_dir, name, 'optimal_net_cls.pth'), map_location='cpu')
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'backbone' in k:
                        new_k = k.replace('backbone1', 'backbone2')
                        new_state_dict[new_k] = v
                self.net_cls.load_state_dict(new_state_dict, strict=False)
                print(name, 'DOPPLER checkpoint loaded')

    def _define_metrics(self, l_state, dataset_mode):
        setattr(self, l_state+'_loss_names', ['c'])
        setattr(self, l_state+'_s_metric_names', ['instance_accuracy'])
        if 'jigsaw' in self.opt.name:
            setattr(self, l_state+'_g_metric_names', ['instance_accuracy'])
        else:
            setattr(self, l_state+'_g_metric_names', ['instance_auc', 'instance_accuracy'])
        setattr(self, l_state+'_t_metric_names', ['instance_cmatrix'])
    
    def get_parameters(self):
        parameter_list = [{"params": self.net_cls.parameters(), "lr_mult": 1, 'decay_mult': 1}]
        return parameter_list

    def set_input(self, data):
        self.input = data['input']
        self.instance_labels = data['instance_label']
        self.input_size = len(self.instance_labels[0])
        self.input_id = data['id'] if isinstance(data['id'], list) else [data['id']+'_{}'.format(i+1) for i in range(self.input_size)]
        
        if self.opt.visualize:
            self.bbox_list = data['bbox_list']
            self.vis_image = data['vis_image']
            
    def forward(self):
        if self.opt.visualize and self.opt.l_state != 'train':
            for visualizer in self.visualizers:
                visualizer.init()

        item_num = len(self.input)
        for i in range(item_num):
            self.input[i] = self.input[i].cuda()
        self.instance_ys = self.net_cls(*tuple(self.input))
        self.instance_scores = []
        self.instance_preds = []

        for i, y in enumerate(self.instance_ys):
            if self.opt.heads[i][1] > 1:
                instance_out = F.softmax(y, dim=1)
                instance_scores = instance_out.detach().cpu()
                self.instance_scores.append(instance_scores)
                if instance_scores.shape[1] == 2 and self.opt.recall_thresholds is not None:
                    instance_preds = instance_scores[:, 1] > self.opt.recall_thresholds[i+1]
                    instance_preds = instance_preds.long()
                else:
                    instance_preds = torch.argmax(instance_out, dim=1)
                self.instance_preds.append(instance_preds)
                
        # calculate the final classification prediction
        dataset = getattr(self, self.opt.l_state+'_dataset')
        target_y, target_score, target_pred = dataset.cal_target(self)
        self.instance_ys.insert(0, target_y)
        self.instance_scores.insert(0, target_score)
        self.instance_preds.insert(0, target_pred)

    def cal_loss(self):
        self.loss_c = 0
        criterion = get_cls_criterion(self.opt.loss_type, self.opt)
        for i, y in enumerate(self.instance_ys):
            self.loss_c += self.opt.loss_weights[i] * criterion(y.cuda(), self.instance_labels[i].long().cuda())

    def stat_info(self):
        super().stat_info()
        
        for i in range(1+len(self.opt.heads)):
            self.buffer_g_instance_scores[i].extend(
                self.instance_scores[i].cpu().tolist())
            self.buffer_g_instance_labels[i].extend(
                self.instance_labels[i].cpu().long().view(-1).tolist())
            self.buffer_g_instance_preds[i].extend(self.instance_preds[i].view(-1).tolist())
        self.buffer_g_ids.extend(self.input_id)

        if self.opt.visualize and self.opt.l_state != 'train':
            self.visualize()

    def save_stat_info(self, epoch):
        super().save_stat_info(epoch)
        records = []
        f = open(os.path.join(self.save_dir, '{}_stat_info_{}.json'.format(epoch, self.opt.v_dataset_id)), 'w')
        for i in range(len(self.buffer_g_ids)):
            sample_idx = self.buffer_g_ids[i]
            instance_labels = [self.buffer_g_instance_labels[j][i] for j in range(1+len(self.opt.heads))]
            instance_scores = [self.buffer_g_instance_scores[j][i] for j in range(1+len(self.opt.heads))]
            record = {'sample_idx': sample_idx,
                        'instance_labels': instance_labels,
                        'instance_scores': instance_scores}
            records.append(record)
        json.dump(records, f)
        f.close()

    def visualize(self):
        self.vis_scores = self.instance_scores[0][:, 1].cpu().numpy()
        all_visualizer_results = []
        for visualizer in self.visualizers:
            results = visualizer.cal_saliency_map(tuple(self.input))
            all_visualizer_results.append(results)
            visualizer.reset()
        self.save_heatmaps(all_visualizer_results)

    def save_heatmaps(self, all_visualizer_results):
        for vis_method, visualizer_result in zip(self.opt.vis_methods, all_visualizer_results):
            for key, saliency_maps in visualizer_result.items():
                for i, saliency_map in enumerate(saliency_maps):
                    np.save(os.path.join(self.vis_dir, '{}_{}_input{}.npy'.format(vis_method, self.input_id[0], i+1)), saliency_map)