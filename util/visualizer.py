import sys
import time
from . import basic
from tensorboardX import SummaryWriter
from util.basic import *
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt, mode='train'):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option

        if opt.l_state == 'train':
            self.writer = SummaryWriter(log_dir='runs/' + opt.name, comment=opt.name)
        else:
            self.writer = SummaryWriter(log_dir='runs/v_' + opt.name, comment='v_' + opt.name)

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, '{}_log.txt'.format(mode))
        self.o_log_p2 = self.log_name.split('/')[-2]
        # with open(self.log_name, "a") as log_file:
        #     now = time.strftime("%c")
        #     log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def plot_current_info(self, model, citer, ptype):
        for k in model.loss_names:
            v = model.meters[k].val
            self.writer.add_scalar(ptype + '/loss_' + k, v, citer)

        for k in model.s_metric_names:
            v = model.meters[k].val
            self.writer.add_scalar(ptype + '/' + k, v, citer)

    def plot_global_info(self, model, citer, ptype):
        if len(model.schedulers) > 0:
            v = model.schedulers[0].lr
            self.writer.add_scalar(ptype + '/lr', v, citer)

        for k in model.loss_names:
            v = model.meters[k].avg
            self.writer.add_scalar(ptype + '/global_loss_' + k, v, citer)

        for k in model.s_metric_names:
            v = model.meters[k].avg
            self.writer.add_scalar(ptype + '/global_' + k, v, citer)

        for k in model.g_metric_names:
            v = model.meters[k].val
            self.writer.add_scalar(ptype + '/' + k, v, citer)

        for k in model.t_metric_names:
            v = model.meters[k].val
            v = numpy2table(v)
            self.writer.add_text(ptype + '/' + k, v, citer)

        model.plot_special_info()

    # losses: same format as |losses| of plot_current_losses
    def print_current_info(self, epoch, iters, model, t_comp, t_data):
        """print current losses on console; also save the losses to the disk


        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, model: %.3fs, data: %.3fs) ' % (epoch, iters, t_comp, t_data)

        if len(model.schedulers) > 0:
            lr_list = model.schedulers[0].get_last_lr()
            v = '['
            for lr in lr_list:
                v += '{:.6f},'.format(lr)
            v = v[:-1] + ']'
            message += '%s: %s ' % ('lr', v)

        for k in model.loss_names:
            v = model.meters[k].val
            message += 'loss_%s: %.3f ' % (k, v)

        for k in model.s_metric_names:
            for i, meter in enumerate(model.meters[k]):
                v = meter.avg
                message += '%s(%d): %.3f ' % (k, i+1, v)

        print(message)  # print the message
        
        # if self.opt.l_state != 'test':
        #     with open(self.log_name, "a") as log_file:
        #         log_file.write('%s\n' % message)  # save the message


    def print_global_info(self, epoch, iters, model, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """

        message = '(epoch: %d, iters: %d, time: %.3fm, data: %.3fm) \n' % (epoch, iters, t_comp, t_data)

        #learning rate

        if len(model.schedulers) > 0:
            lr_list = model.schedulers[0].get_last_lr()
            v = '['
            for lr in lr_list:
                v += '{:.6f},'.format(lr)
            v = v[:-1] + ']'
            message += '%s: %s ' % ('lr', v)

        for k in model.loss_names:
            v = model.meters[k].avg
            message += 'loss_%s: %.3f ' % (k, v)

        for k in model.s_metric_names:
            for i, meter in enumerate(model.meters[k]):
                v = meter.avg
                message += '%s(%d): %.3f ' % (k, i+1, v)

        for k in model.g_metric_names:
            for i, meter in enumerate(model.meters[k]):
                v = meter.avg
                message += '%s(%d): %.3f ' % (k, i+1, v)

        for k in model.t_metric_names:
            for i, meter in enumerate(model.meters[k]):
                v = meter.sum
                message += '%s(%d): \n %s\n' % (k, i+1, str(v))

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

        # save roc curve
        if model.opt.save_roc_dir is not None:
            save_dir = os.path.join(model.opt.save_roc_dir, model.opt.name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
        model.print_special_info(self.log_name)


    def visual_images(self,model):
        save_path = self.opt.image_save_dir + '/' + self.opt.name

        mkdir(save_path)

        if 'ac' in self.opt.plot_info:
            model.visualization()

        v_type_list = []
        if 'right' in self.opt.plot_info or 'correct' in self.opt.plot_info:
            v_type_list.append('correct')

        if 'wrong' in self.opt.plot_info:
            v_type_list.append('wrong')

        for v_type in v_type_list:
            im_list = model.get_v_images(v_type)
            for label,im in enumerate(im_list):
                if not isinstance(im,type(None)):
                    im = np.repeat(np.transpose(im,((1,2,0))),3,-1)
                    cv2.imwrite(save_path + '/' + v_type + '_' + str(label) + '.jpg',im)
                    print('im shape ' + str(im.shape))
