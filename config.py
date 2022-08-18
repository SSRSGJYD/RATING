####################################### Configuration ############################################
test_dataset = './dataset_files/test.json'
GS_checkpoints = './checkpoints/GS/'
DOPPLER_checkpoints = './checkpoints/DOPPLER/'
GSDOPPLER_checkpoints = './checkpoints/GSDOPPLER/'
save_dir = './output'
save_heatmap = False

GLOBAL_DICT = dict()
kfold = 5
train_repeat = 2
GLOBAL_DICT['GS_dir'] = GS_checkpoints
GLOBAL_DICT['DOPPLER_dir'] = DOPPLER_checkpoints
GLOBAL_DICT['GSDOPPLER_dir'] = GSDOPPLER_checkpoints

arch = 18
GLOBAL_DICT['GSDOPPLER_SH_CONDITION'] = "'label=SH' in x and 'archs={}' in x and 'frozen_stages=4' in x and 'in_channels=3,3' in x".format(arch)
GLOBAL_DICT['DOPPLER_VASCULARITY_CONDITION'] = "'label=vascularity' in x and 'archs={}' in x and 'frozen_stages=0' in x".format(arch)

import os
GLOBAL_DICT['DOPPLER_names'] = os.listdir(GLOBAL_DICT['DOPPLER_dir'])
GLOBAL_DICT['GSDOPPLER_names'] = os.listdir(GLOBAL_DICT['GSDOPPLER_dir'])