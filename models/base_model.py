from abc import ABC, abstractmethod
from collections import OrderedDict
import json
import os
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
from schedulers import get_scheduler, metric_schedulers
from util import *


class BaseModel(object):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.wait_over = False
        self.start_forward = True
        self.wait_epoch = 0

        self.gpu_ids = opt.gpu_ids
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.o_save_dir = self.save_dir

        self._define_metrics('train', self.opt.dataset_mode)
        self._define_metrics('valid', self.opt.v_dataset_mode)
        self._define_metrics('test', self.opt.v_dataset_mode)

        self.net_names = []
        self.optimizers = []

        self.best_m_value = -2
        self.c_grad_iter = 0

        self._define_buffers()

    @abstractmethod
    def _define_metrics(self, l_state, dataset_mode):
        pass

    @abstractmethod
    def _define_buffers(self):
        pass

    @staticmethod
    def modify_commandline_options(parser):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.
        """

        # model
        parser.add_argument('--method_name', type=str, default='GSDopplerFeatureFusion')
        parser.add_argument('--layer_norm_type', type=str, default='batch', choices=('batch','instance','group','frn'))
        parser.add_argument('--activation_type', type=str, default='relu', choices=('relu','prelu','leaky_relu','tanh'))
        
        # training
        parser.add_argument('--reinit_data', type=int, default=0)

        return parser

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        self.buffer_names = self.get_buffer_names()
        self.meters = self.gen_meters()
        self.schedulers = []

        def _get_device_index(device, optional=False):
            r"""Gets the device index from :attr:`device`, which can be a torch.device
            object, a Python integer, or ``None``.

            If :attr:`device` is a torch.device object, returns the device index if it
            is a CUDA device. Note that for a CUDA device without a specified index,
            i.e., ``torch.device('cuda')``, this will return the current default CUDA
            device if :attr:`optional` is ``True``.

            If :attr:`device` is a Python integer, it is returned as is.

            If :attr:`device` is ``None``, this will return the current default CUDA
            device if :attr:`optional` is ``True``.
            """
            if isinstance(device, torch._six.string_classes):
                device = torch.device(device)
            if isinstance(device, torch.device):
                dev_type = device.type
                if device.type != 'cuda':
                    raise ValueError('Expected a cuda device, but got: {}'.format(device))
                device_idx = device.index
            else:
                device_idx = device
            if device_idx is None:
                if optional:
                    # default cuda device index
                    return torch.cuda.current_device()
                else:
                    raise ValueError('Expected a cuda device with a specified index '
                                    'or an integer, but got: '.format(device))
            return device_idx

        self.gpu_num = len(self.opt.gpu_ids)
        self.device_ids = list(map(lambda x: _get_device_index(x, True), self.opt.gpu_ids))

        self.set_l_state(self.opt.l_state)
        if opt.l_state == 'train':
            self.set_optimizer(opt)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            # updated for pytorch 1.9
            for scheduler in self.schedulers:
                scheduler.step()

        # load checkpoints
        if opt.l_state == 'train':
            if opt.load_dir is not None:
                load_suffix = 'optimal'
                self.load_networks(load_suffix, load_dir=opt.load_dir)
        else:
            if opt.load_dir is not None:
                load_suffix = 'optimal'
                self.load_networks(load_suffix, load_dir=opt.load_dir)
            else:
                load_suffix = 'optimal'
                self.load_networks(load_suffix)
                self.print_networks(opt.verbose)

        # save base model
        # for name in self.net_names:
        #     if isinstance(name, str):
        #         save_filename = '%s_net_%s.pth' % ('base', name)
        #         save_path = os.path.join(self.save_dir, save_filename)
        #         net = getattr(self, 'net_' + name)
        #         torch.save(net.state_dict(), save_path)

        for name in self.net_names:
            net = getattr(self, 'net_' + name)
            # net.cuda()
            net.cuda(self.gpu_ids[0])
            if self.opt.parallel_mode == 'dataparallel':
                setattr(self, 'net_' + name, nn.DataParallel(net, opt.gpu_ids))
            else:
                setattr(self, 'net_' + name, nn.parallel.DistributedDataParallel(net, opt.gpu_ids))
            if self.opt.l_state == 'train':
                net.train()
            else:
                net.eval()

    def set_l_state(self, l_state):
        assert l_state in ['train', 'valid', 'test']
        self.opt.l_state = l_state
        self.loss_names = getattr(self, l_state+'_loss_names')
        self.s_metric_names = getattr(self, l_state+'_s_metric_names')
        self.g_metric_names = getattr(self, l_state+'_g_metric_names')
        self.t_metric_names = getattr(self, l_state+'_t_metric_names')
        if self.opt.l_state == 'train':
            self.current_dataset_mode = self.opt.dataset_mode
        else:
            self.current_dataset_mode = self.opt.v_dataset_mode

    def gen_meters(self):
        meters = {}
        if self.opt.l_state in ['train', 'valid']:
            valid_states = ['train', 'valid']
        else:
            valid_states = ['test']

        for ntype in ['loss']:
            for l_state in valid_states:
                name_list = getattr(self, l_state + '_' + ntype + '_names')
                for name in name_list:
                    meters[name] = metrics.Meter()
        
        name_types = ['s_metric', 't_metric', 'g_metric']
        for ntype in name_types:
            for l_state in valid_states:
                name_list = getattr(self, l_state + '_' + ntype + '_names')
                for name in name_list:
                    if hasattr(self.opt, 'heads'):
                        meters[name] = [metrics.Meter() for i in range(1+len(self.opt.heads))]
                    else:
                        meters[name] = [metrics.Meter()]
        return meters

    def update_metrics(self, m_type = 'local'):
        if not self.start_forward and m_type != 'global':
            return
        if m_type == 'global':
            name_types = ['t_metric', 'g_metric']
        else:
            name_types = ['loss', 's_metric']

        for ntype in name_types:
            cal_func = getattr(self,'cal_' + ntype)
            cal_func()
            name_list = getattr(self, ntype + '_names')
            for name in name_list:
                self.update_meters(ntype, name)

    def update_meters(self, ntype, name):
        value_list = getattr(self, ntype + '_' + name)

        if isinstance(value_list, list):
            for i, value in enumerate(value_list):
                if isinstance(value,torch.Tensor):
                    value = value.detach().cpu().numpy()

                if isinstance(value, np.ndarray) and ntype != 't_metric':
                    value = value.item()

                if ntype != 't_metric':
                    self.meters[name][i].update(value, self.input_size)
                else:
                    self.meters[name][i].update(value, 1)
        else:
            value = value_list
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()

            if isinstance(value, np.ndarray) and ntype != 't_metric':
                value = value.item()

            if ntype != 't_metric':
                self.meters[name].update(value, self.input_size)
            else:
                self.meters[name].update(value, 1)

    def reset_meters(self):
        name_list = getattr(self, 'loss_names')
        for name in name_list:
            self.meters[name].reset()

        name_types = ['s_metric', 't_metric', 'g_metric']
        for ntype in name_types:
            name_list = getattr(self, ntype + '_names')
            for name in name_list:
                # value = getattr(self, ntype + '_' + name)
                for meter in self.meters[name]:
                    meter.reset()

    @abstractmethod
    def get_parameters(self):
        pass

    def clear_info(self):
        # print(' buffer name ' + str(self.buffer_names))
        for name in self.buffer_names:
            if name == 'names':
                continue
            tmp_buffer = getattr(self,'buffer_' + name)
            if len(tmp_buffer) > 0:
                if isinstance(tmp_buffer[0],list):
                    tmp_buffer = [[] for _ in range(len(tmp_buffer))]
                else:
                    tmp_buffer = []
            setattr(self,'buffer_' + name, tmp_buffer)
            # value = getattr(self, 'buffer_' + name)

    def set_optimizer(self,opt):
        if opt.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.get_parameters(), lr=opt.lr, momentum=opt.momentum, nesterov=opt.nesterov,
                                             weight_decay=opt.weight_decay)
        elif opt.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.get_parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay = opt.weight_decay)
        elif opt.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.get_parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay = opt.weight_decay)
        
        # updated for pytorch 1.9
        self.optimizer.zero_grad()
        self.optimizer.step()

        self.optimizers = [self.optimizer]

    @abstractmethod
    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        pass

    def clip_grad_norm_(self, max_norm, norm_type=2):
        r"""Clips gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """

        parameters = self.get_parameters()

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if norm_type == 'inf':
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            total_norm = 0
            for p in parameters:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
        return total_norm

    def validate_parameters(self):
        self.start_forward = True

        if self.opt.visualize:
            self.forward()
        else:
            with torch.no_grad():
                self.forward()
        # self.backward()
        self.stat_info()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.start_forward = True
        self.forward()  # first call forward to calculate intermediate results
        self.backward()  # calculate gradients for network G
        self.stat_info()
        self.c_grad_iter += 1

        if self.c_grad_iter == self.opt.grad_iter_size:
            self.optimizer.step()  # update gradients for network G
            self.optimizer.zero_grad()  # clear network G's existing gradients
            self.c_grad_iter = 0

    @abstractmethod
    def stat_info(self):
        pass

    @abstractmethod
    def save_stat_info(self, epoch):
        pass

    def get_buffer_names(self):
        v_names = list(self.__dict__.keys())
        b_names = [v.replace('buffer_','') for v in v_names if v.startswith('buffer')]
        # print(b_names)
        return b_names

    def zero_grad(self):
        for n_name in self.net_names:
            net = getattr(self,'net_' + n_name)
            net.zero_grad()

    @abstractmethod
    def cal_loss(self):
        pass

    # def accuracy(self, score, label):
    #     if self.opt.recall_thresh is not None:
    #         threshold = self.opt.recall_thresh
    #     else:
    #         threshold = 0.5
    #     pred = score > threshold
    #     pred = pred.cpu().long()
    #     correct = (label.cpu().long() == pred)
    #     return 100.0 * correct.sum() / len(correct)

    def cal_s_metric(self):
        if 'bag_accuracy' in self.s_metric_names:
            self.s_metric_bag_accuracy = metrics.accuracy(
                self.bag_preds, self.bag_label)
        if 'instance_accuracy' in self.s_metric_names:
            self.s_metric_instance_accuracy = [metrics.accuracy(
                self.instance_preds[i], self.instance_labels[i]) for i in range(1+len(self.opt.heads))]

    def cal_g_metric(self):
        if 'bag_precision' in self.g_metric_names:
            self.g_metric_bag_precision = metrics.precision(
                self.t_metric_bag_cmatrix, 1)
        if 'bag_recall' in self.g_metric_names:
            self.g_metric_bag_recall = metrics.recall(
                self.t_metric_bag_cmatrix, 1)
        if 'bag_fscore' in self.g_metric_names:
            self.g_metric_bag_fscore = metrics.f_score(
                self.t_metric_bag_cmatrix, 1)
        if 'bag_auc' in self.g_metric_names:
            self.g_metric_bag_auc = metrics.auc_score(
                self.buffer_g_bag_labels, self.buffer_g_bag_scores)
            self.bag_fpr, self.bag_tpr, self.bag_thresholds = roc_curve(
                self.buffer_g_bag_labels, self.buffer_g_bag_scores)
        if 'instance_precision' in self.g_metric_names:
            self.g_metric_instance_precision = [metrics.precision(
                self.t_metric_instance_cmatrix[i], 1) for i in range(1+len(self.opt.heads))]
        if 'instance_recall' in self.g_metric_names:
            self.g_metric_instance_recall = [metrics.recall(
                self.t_metric_instance_cmatrix[i], 1) for i in range(1+len(self.opt.heads))]
        if 'instance_fscore' in self.g_metric_names:
            self.g_metric_instance_fscore = [metrics.f_score(
                self.t_metric_instance_cmatrix[i], 1) for i in range(1+len(self.opt.heads))]
        if 'instance_auc' in self.g_metric_names:
            self.g_metric_instance_auc = [metrics.auc_score(
                self.buffer_g_instance_labels[i], self.buffer_g_instance_scores[i]) for i in range(1+len(self.opt.heads))]
            # self.instance_fpr, self.instance_tpr, self.instance_thresholds = [roc_curve(
            #     self.buffer_g_instance_labels[i], self.buffer_g_instance_scores[i]) for i in range(1+len(self.opt.heads))]
        if 'instance_accuracy' in self.g_metric_names:
            self.g_metric_instance_accuracy = [metrics.accuracy_from_matrix(
                self.t_metric_instance_cmatrix[i]) for i in range(1+len(self.opt.heads))]

    def cal_t_metric(self):
        if 'bag_cmatrix' in self.t_metric_names:
            self.t_metric_bag_cmatrix = [metrics.comfusion_matrix(
                self.buffer_g_bag_preds[i], self.buffer_g_bag_labels[i], self.opt.heads[i][1]) for i in range(1+len(self.opt.heads))]
        if 'instance_cmatrix' in self.t_metric_names:
            class_nums = [head[1] for head in self.opt.heads]
            class_nums.insert(0, len(self.opt.classes[0]))
            self.t_metric_instance_cmatrix = [metrics.comfusion_matrix(
                self.buffer_g_instance_preds[i], self.buffer_g_instance_labels[i], class_nums[i]) for i in range(1+len(self.opt.heads))]

    def backward(self):
        self.update_metrics('local')
        total_loss = 0
        for name in self.loss_names:
            loss = getattr(self,'loss_' + name) / self.opt.grad_iter_size
            total_loss += loss
        total_loss.backward()

    def validation(self, dataset, visualizer, valid_iter, epoch):
        self.eval()
        self.set_l_state('valid')
        
        iter_time_meter = metrics.TimeMeter()
        data_time_meter = metrics.TimeMeter()

        data_time_meter.start()
        iter_time_meter.start()

        for i, data in enumerate(dataset):  # inner loop within one epoch
            data_time_meter.record(n = self.opt.batch_size)
            iter_time_meter.start()
            self.set_input(data)
            self.validate_parameters()
            self.update_metrics('local')

            iter_time_meter.record()

            if i % self.opt.v_print_freq == 0:  # print training losses and save logging information to the disk
                visualizer.print_current_info(-1, i, self, iter_time_meter.val, data_time_meter.val)

            data_time_meter.start()
            iter_time_meter.start()

        self.update_metrics('global')
        visualizer.print_global_info(-1, -1, self, iter_time_meter.sum/60, data_time_meter.sum/60)
        # visualizer.plot_global_info(self, valid_iter, ptype='valid')
        
        tmp_v_value = self.get_metric(self.opt.valid_metric)
        if (self.opt.greater_is_better and tmp_v_value >= self.best_m_value) or ((not self.opt.greater_is_better) and tmp_v_value <= self.best_m_value):
            self.best_epoch_metrics = 'Best model: ' + self.get_all_metrics_message()
            self.save_stat_info('optimal_valid')

        self.reset_meters()
        self.clear_info()
        self.train()
        self.set_l_state('train')

    def test(self, dataset, visualizer, valid_iter, epoch):
        self.eval()
        self.set_l_state('test')

        iter_time_meter = metrics.TimeMeter()
        data_time_meter = metrics.TimeMeter()

        data_time_meter.start()
        iter_time_meter.start()

        for i, data in enumerate(dataset):  # inner loop within one epoch
            data_time_meter.record(n = self.opt.batch_size)
            iter_time_meter.start()
            self.set_input(data)
            self.validate_parameters()
            self.update_metrics('local')

            iter_time_meter.record()

            if i % self.opt.v_print_freq == 0:  # print training losses and save logging information to the disk
                visualizer.print_current_info(-1, i, self, iter_time_meter.val, data_time_meter.val)

            data_time_meter.start()
            iter_time_meter.start()

        self.update_metrics('global')
        visualizer.print_global_info(-1, -1, self, iter_time_meter.sum/60, data_time_meter.sum/60)
        # visualizer.plot_global_info(self, valid_iter, ptype='valid')
        
        self.save_stat_info(epoch)

        self.reset_meters()
        self.clear_info()

    def plot_special_info(self):
        pass

    def print_special_info(self,log_name):
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.net_names:
            net = getattr(self, 'net_' + name)
            net.eval()

    def train(self):
        for name in self.net_names:
            net = getattr(self, 'net_' + name)
            net.train()

    # def test(self):
    #     """Forward function used in test time.

    #     This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
    #     It also calls <compute_visuals> to produce additional visualization results
    #     """
    #     if self.opt.visualize:
    #         self.forward()
    #         self.compute_visuals()
    #     else:
    #         with torch.no_grad():
    #             self.forward()
    #             self.compute_visuals()

    def get_metric(self,metric_name,all=False):
        if all:
            try:
                value = [float(x) for x in getattr(self, 'g_metric_' + metric_name)]
            except:
                value = [float(x) for x in getattr(self, 's_metric_' + metric_name)]
        else:
            try:
                value = float(getattr(self, 'g_metric_' + metric_name)[0])
            except:
                value = float(getattr(self, 's_metric_' + metric_name)[0])
        return value
    
    def get_all_metrics_message(self):
        message = ''
        for k in self.loss_names:
            v = self.meters[k].avg
            message += 'loss_%s: %.3f ' % (k, v)

        for k in self.s_metric_names:
            for i, meter in enumerate(self.meters[k]):
                v = meter.avg
                message += '%s(%d): %.3f ' % (k, i+1, v)

        for k in self.g_metric_names:
            for i, meter in enumerate(self.meters[k]):
                v = meter.avg
                message += '%s(%d): %.3f ' % (k, i+1, v)

        for k in self.t_metric_names:
            for i, meter in enumerate(self.meters[k]):
                v = meter.sum
                message += '%s(%d): \n %s\n' % (k, i+1, str(v))
        
        return message

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy in metric_schedulers:
                tmp_metric = self.get_metric(self.opt.scheduler_metric)
                m_kind = self.get_metric_kind(self.opt.scheduler_metric)
                if m_kind == 'loss' or not self.opt.greater_is_better:
                    tmp_metric = -tmp_metric
                scheduler.step(metrics=tmp_metric)
            else:
                scheduler.step()

    def save_optimal_networks(self, visualizer):
        """Save all the networks to the disk.
        """
        print('valid metric ' + str(self.opt.valid_metric))
        tmp_v_value = self.get_metric(self.opt.valid_metric)
        print('v value ' + str(tmp_v_value))
        if (self.opt.greater_is_better and tmp_v_value >= self.best_m_value) or ((not self.opt.greater_is_better) and tmp_v_value <= self.best_m_value):
            self.best_m_value = tmp_v_value
            old_save_dir_sh = self.save_dir.replace('(','\(').replace(')','\)')
            self.save_dir = self.o_save_dir + '({}={:.3f})'.format(self.opt.valid_metric, self.best_m_value)
            new_save_dir_sh = self.save_dir.replace('(','\(').replace(')','\)')
            os.system('mv ' + old_save_dir_sh + ' ' + new_save_dir_sh)

            pred_fname = os.path.join(new_save_dir_sh, 'pred_result.txt')
            if os.path.exists(pred_fname):
                n_pred_fname = pred_fname.replace('pred','optimal_pred')
                os.system('mv ' + pred_fname + ' ' + n_pred_fname)

            # print('log name ' + visualizer.log_name)
            log_parts = visualizer.log_name.split('/')
            log_parts[-2] = visualizer.o_log_p2 + '({}={:.3f})'.format(self.opt.valid_metric, self.best_m_value)
            visualizer.log_name = ''
            for p in log_parts:
                visualizer.log_name += p + '/'
            visualizer.log_name = visualizer.log_name[:-1]

            self.wait_epoch = 0

            for name in self.net_names:
                if isinstance(name, str):
                    save_filename = 'optimal_net_%s.pth' % (name)
                    self.save_path = os.path.join(self.save_dir, save_filename)
                    net = getattr(self, 'net_' + name)

                    if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                        # torch.save(net.module.cpu().state_dict(), self.save_path)
                        torch.save(net.module.state_dict(), self.save_path)
                        # net.cuda(self.gpu_ids[0])
                    else:
                        torch.save(net.state_dict(), self.save_path)

        else:
            self.wait_epoch += 1
            if self.wait_epoch > self.opt.patient_epoch:
                self.wait_over = True

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.net_names:
            if isinstance(name, str):
                save_filename = '{}_net_{}.pth'.format(epoch, name)
                self.save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net_' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.state_dict(), self.save_path)
                else:
                    torch.save(net.state_dict(), self.save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch, load_dir = None):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.net_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if load_dir is None:
                    load_dir = self.save_dir
                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, 'net_' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                elif isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location='cpu')
                net.load_state_dict(state_dict, strict=False)
                print('model loaded')

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.net_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        
    def next_epoch(self):
        if self.opt.reinit_data:
            self.opt.dataset.dataset.__init__(self.opt, 'train')

    def get_metric_kind(self, m_name):
        if m_name in self.loss_names:
            return 'loss'
        elif m_name in self.s_metric_names:
            return 's_metric'
        elif m_name in self.t_metric_names:
            return 't_metric'
        elif m_name in self.g_metric_names:
            return 'g_metric'

        AssertionError(False, 'This metric is not in this model')
