from util.basic import *
from util.metrics import TimeMeter
from models import create_model
from options.base_options import TrainOptions
from util.visualizer import Visualizer

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import find_dataset_using_name
from datasets.jigsaw_puzzle import JigsawPuzzle

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    opt.milestones = []

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True

    original_train_dataset = find_dataset_using_name(opt.dataset_mode)(opt, 'train')
    jigsaw_train_dataset = JigsawPuzzle(opt, 'train', original_train_dataset, permutation_num=100, permutation_file='./datasets/permutations_1000.npy', width_crops=3, height_crops=3, patch_width=64, patch_height=64)
    collate_fn = jigsaw_train_dataset.get_collate_fn() if hasattr(jigsaw_train_dataset, 'collate_fn') else None
    jigsaw_train_dataloader = DataLoader(
                                jigsaw_train_dataset,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=int(opt.num_threads),
                                collate_fn=collate_fn,
                                pin_memory=False,
                                drop_last=False)
    print('The number of training samples = %d' % len(jigsaw_train_dataset))

    if opt.valid_model:
        original_valid_dataset = find_dataset_using_name(opt.dataset_mode)(opt, 'valid')
        jigsaw_valid_dataset = JigsawPuzzle(opt, 'valid', original_valid_dataset, permutation_num=100, permutation_file='./datasets/permutations_1000.npy', width_crops=3, height_crops=3, patch_width=64, patch_height=64)
        collate_fn = jigsaw_valid_dataset.get_collate_fn() if hasattr(jigsaw_valid_dataset, 'collate_fn') else None
        jigsaw_valid_dataloader = DataLoader(
                                    jigsaw_valid_dataset,
                                    batch_size=opt.batch_size,
                                    shuffle=True,
                                    num_workers=int(opt.num_threads),
                                    collate_fn=collate_fn,
                                    pin_memory=False,
                                    drop_last=False)
        print('The number of validation samples = %d' % len(jigsaw_valid_dataset))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.train_dataset = original_train_dataset
    model.valid_dataset = original_valid_dataset

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    valid_iters = 0
    valid_freq = opt.valid_freq

    iter_time_meter = TimeMeter()
    data_time_meter = TimeMeter()
    epoch_time_meter = TimeMeter()

    print('Start to train')
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        original_train_dataset.prepare_new_epoch()
        if opt.single_valid_freq_epoch is not None and epoch >= opt.single_valid_freq_epoch:
            valid_freq = len(jigsaw_train_dataset) // opt.batch_size

        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_time_meter.start()  # timer for entire epoch
        data_time_meter.start()
        iter_time_meter.start()
        for i, data in enumerate(jigsaw_train_dataloader):  # inner loop within one epoch
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
                        model.validation(jigsaw_valid_dataloader, visualizer, valid_iters, epoch)
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