import os
from options.base_options import TestOptions
from datasets import create_dataset
from models import create_model
from util.visualizer import Visualizer
from config import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    for target in ['GSDOPPLER_SH', 'DOPPLER_VASCULARITY']:
        opt = TestOptions(target.split('_')[0]).parse()  # get test options
        opt.gpu_ids = [0]
        opt.model = 'cls'
        opt.method_name = 'GSDopplerFeatureFusion'
        opt.policy = 'roi'
        opt.checkpoints_dir = GLOBAL_DICT[target.split('_')[0]+'_dir']
        opt.remark = ''

        if target == 'GSDOPPLER_SH':
            opt.backbones = ['resnet', 'resnet']
            opt.archs = [str(arch), str(arch)]
            opt.heads = [([0,1], 2)]
            opt.in_channels = [3, 3]
            opt.label = 'SH'
        elif target == 'DOPPLER_VASCULARITY':
            opt.backbones = ['resnet']
            opt.archs = [str(arch)]
            opt.heads = [([0], 2)]
            opt.in_channels = [3]
            opt.label = 'vascularity'
        
        all_classes = ['0,123-0,123', '01,23-01,23', '3,012-3,012']
        for classes in all_classes:
            opt.classes = classes.split('-')
            opt.classes = [s.split(',') for s in opt.classes]
            if save_heatmap and target == 'GSDOPPLER_SH' and classes == '0,123-0,123':
                opt.visualize = 1
                opt.vis_methods = ['IG']
                opt.vis_layer_names = ['layer4']
                opt.v_batch_size = 1

            for split in range(1, kfold+1):
                for seed in range(0, train_repeat):
                    opt.l_state = 'test'
                    opt.name = list(filter(lambda x:eval(GLOBAL_DICT[target+'_CONDITION']) and 'classes={}'.format(classes) in x and 'split={}'.format(split) in x and 'seed={}'.format(seed) in x, GLOBAL_DICT[target.split('_')[0]+'_names']))[0]
                    print(opt.name)
                    opt.load_dir = os.path.join(opt.checkpoints_dir, opt.name)
                    opt.v_dataset_id = 0                    
                    opt.test_datasets = [[]] * opt.v_dataset_id + [['"{}"'.format(test_dataset)]]

                    visualizer = Visualizer(opt, opt.l_state)
                    t_dataset = create_dataset(opt, opt.l_state)

                    model = create_model(opt)
                    model.setup(opt)
                    model.test_dataset = t_dataset.dataset
                    model.save_dir = os.path.join(save_dir, target, opt.name)
                    if not os.path.exists(model.save_dir):
                        os.makedirs(model.save_dir)
                    model.test(t_dataset, visualizer, -1, 'optimal_test')