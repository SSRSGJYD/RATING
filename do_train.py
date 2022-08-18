from util.basic import *
from util.metrics import TimeMeter
from models import create_model
from options.base_options import TrainOptions
from datasets import create_dataset
from util.visualizer import Visualizer
import shutil

import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True

    dataset = create_dataset(opt, 'train')  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training samples = %d' % dataset_size)

    if opt.valid_model:
        v_dataset = create_dataset(opt, 'valid')
        print('The number of validation samples = %d' % len(v_dataset))

    if opt.test_model:
        t_datasets = []
        for i, datasets in enumerate(opt.test_datasets):
            opt.v_dataset_id = i
            t_dataset = create_dataset(opt, 'test')
            print('The number of test samples = %d' % len(t_dataset))
            t_datasets.append(t_dataset)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.train_dataset = dataset.dataset
    model.valid_dataset = v_dataset.dataset

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    valid_iters = 0
    valid_freq = opt.valid_freq

    iter_time_meter = TimeMeter()
    data_time_meter = TimeMeter()
    epoch_time_meter = TimeMeter()

    print('Start to train')

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        dataset.dataset.prepare_new_epoch()
        if opt.single_valid_freq_epoch is not None and epoch >= opt.single_valid_freq_epoch:
            valid_freq = len(dataset) // opt.batch_size

        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_time_meter.start()  # timer for entire epoch
        data_time_meter.start()
        iter_time_meter.start()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            data_time_meter.record()

            iter_time_meter.start()
            visualizer.reset()
            total_iters += 1
            epoch_iter += 1
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            iter_time_meter.record()

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                visualizer.print_current_info(epoch, epoch_iter, model, iter_time_meter.val, data_time_meter.val)

            if total_iters % valid_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                model.update_metrics('global')
                visualizer.print_global_info(epoch, epoch_iter, model, iter_time_meter.sum/60,data_time_meter.sum/60)

                iter_time_meter.reset()
                data_time_meter.reset()

                model.reset_meters()
                model.clear_info()
                if opt.valid_model:
                    with torch.no_grad():
                        model.validation(v_dataset, visualizer, valid_iters, epoch)
                model.update_learning_rate()

                model.save_optimal_networks(visualizer)
                if model.wait_over:
                    break

                model.reset_meters()
                valid_iters += 1

            data_time_meter.start()
            iter_time_meter.start()

        if model.wait_over:
            print('early stop at %d / %d' % (epoch,epoch_iter))
            break

        epoch_time_meter.record()
        epoch_time_meter.start()
        
        model.next_epoch()

        print('End of epoch %d / %d \t Time Taken: %d hours' % (epoch, opt.niter + opt.niter_decay, epoch_time_meter.sum/3600.))

    print(model.best_epoch_metrics)
    with open(visualizer.log_name, "a") as log_file:
        log_file.write('%s\n' % model.best_epoch_metrics)  # save the message

    if opt.test_model:
        opt.phase = 'test'
        print('test model')

        # test optimal model
        model.reset_meters()
        model.clear_info()
        model.load_networks('optimal', model.save_dir)
        t_values = []
        for i, t_dataset in enumerate(t_datasets):
            opt.v_dataset_id = i
            model.test_dataset = t_dataset.dataset
            with torch.no_grad():
                model.test(t_dataset, visualizer, -1, 'optimal_test')
            t_value = model.get_metric(model.opt.valid_metric)
            t_values.append(t_value)
        ind = -1
        while model.save_dir[ind] != ')':
            ind -= 1
        new_save_dir = model.save_dir[:ind]
        for t_value in t_values:
            new_save_dir += ',{:.3f}'.format(t_value)

        new_save_dir += ')'
        shutil.move(model.save_dir, new_save_dir)
        model.save_dir = new_save_dir