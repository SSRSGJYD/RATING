from util.basic import *
import os
import datetime
import setting
from config import *
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

gpu_list = '0'  # gpus to runs The element of the list is 'g0,g1,...g,n-1',where n is number of gpu each task uses
setting_name = 'SH_classifier'
kfold = 5
train_repeat = 2
param_setting = setting.get_param_setting(setting_name)
start_code = ''
set_visible_device = True

def getTime():
    timeNow = datetime.datetime.now().strftime('%b%d_%H-%M')
    return timeNow

def genetate_code(tmp_code, tmp_text, code_list, text_list, params_list, value_matrix, arg_dict):
    tmp_params = params_list.split()
    p_code = tmp_code
    p_text = tmp_text
    for i in range(len(tmp_params)):
        p_code = p_code + ' --' + tmp_params[i] + ' ' + str(value_matrix[i])
        p_text = p_text + ',' + tmp_params[i] + '=' + str(value_matrix[i])
    code_list.append(p_code)
    p_text = getTime() + '_' + p_text
    for k, v in arg_dict.items():
        p_text = p_text + ',{}={}'.format(k, v)
    text_list.append(p_text)

if __name__ == "__main__":
    gpu_num = len(gpu_list.split(','))
    add_info = setting_name

    params_list = param_setting.params['params']
    value_matrix = param_setting.params['values']
    task = param_setting.params['values'][0]
    label = param_setting.params['values'][1]
    
    for split in range(1, kfold+1):
        for classes in ['0,123-0,123', '01,23-01,23', '3,012-3,012']:
            for expid in [0]:
                    for seed in range(0, train_repeat):
                        code_list = []
                        text_list = []
                        arg_dict = {
                            'classes': classes,
                            'split': split,
                            'seed': seed
                        }
                        genetate_code(start_code,add_info,code_list,text_list,params_list,value_matrix,arg_dict)

                        bash = ''
                        if set_visible_device:
                            bash += 'CUDA_VISIBLE_DEVICES=' + str(gpu_list)
                        bash += ' python do_train.py ' + code_list[0]

                        tmp_gpus = ''
                        for j in range(gpu_num):
                            tmp_gpus += str(j) + ','
                        tmp_gpus = tmp_gpus[:-1]            
                        
                        bash += ' --gpu_ids ' + tmp_gpus
                        bash += ' --name ' + str(text_list[0])    
                        bash += ' --classes {} --split {} --seed {}'.format(classes, split, seed)
                        
                        os.system(bash)