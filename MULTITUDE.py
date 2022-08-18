import json
import numpy as np
import os
import pickle

from config import *
import statistic_util

KEY_DICT = {
    '0,123-0,123': [0, 1, 1, 1], 
    '01,23-01,23': [0, 0, 1, 1],
    '3,012-3,012': [1, 1, 1, 0]
}

def L1_distance(query, key, weight):
    return np.sum(np.abs(query - key) * weight)

def ECOC(preds, keys, weight):
    '''
    preds: (6, N)
    keys: (6, 4)
    weight: (6)
    '''
    ECOC_preds = []
    querys = np.array(preds)
    for i in range(len(preds[0])):
        distances = [L1_distance(querys[:, i], key, weight) for key in keys]
        pred = np.argmin(distances) # if same, return smallest index
        ECOC_preds.append(pred)
    return ECOC_preds

def single_model_prediction(target):
    binary_combination = ['0,123-0,123', '01,23-01,23', '3,012-3,012']
    keys = np.stack([KEY_DICT[classes] for classes in binary_combination] * train_repeat, axis=1)
    weight = np.array([1,1,1] * train_repeat)
    
    all_preds = []
    for split in range(1, kfold+1):
        preds = []
        for seed in range(0, train_repeat):
            for classes in binary_combination:
                name = list(filter(lambda x:eval(GLOBAL_DICT[target+'_CONDITION']) and 'classes={}'.format(classes) in x and 'expid={}'.format(0) in x and 'split={}'.format(split) in x and 'seed={}'.format(seed) in x, GLOBAL_DICT[target.split('_')[0]+'_names']))[0]                
                with open(os.path.join(save_dir, target, name, 'optimal_test_stat_info_0.json'), 'r') as f:
                    results = json.load(f)
                scores = [result['instance_scores'][0][1] for result in results]
                thresh = GLOBAL_DICT[target+'_THRESH_DICT'][classes][0][split][seed]
                pred = [1 if score >= thresh else 0 for score in scores]
                preds.append(pred)

        combined_preds = ECOC(preds, keys, weight)
        all_preds.append(combined_preds)

    return all_preds

def do_MULTITUDE(all_SH_preds, all_VASCULARITY_preds):
    SH_preds = []
    VASCULARITY_preds = []
    combined_preds = []
    N = len(all_SH_preds[0])
    for i in range(N):
        SH_counts = np.bincount([all_SH_preds[j][i] for j in range(5)])
        VASCULARITY_counts = np.bincount([all_VASCULARITY_preds[j][i] for j in range(5)])

        SH_pred, VASCULARITY_pred = None, None
        maximum = 0
        for j in range(len(SH_counts)):
            if SH_counts[j] == 0: continue
            for k in range(len(VASCULARITY_counts)):
                if VASCULARITY_counts[k] == 0: continue
                if not (j == 0 and k > 0) and SH_counts[j] * VASCULARITY_counts[k] > maximum:
                    SH_pred, VASCULARITY_pred = j, k
                    maximum = SH_counts[j] * VASCULARITY_counts[k]
        if SH_pred is None and VASCULARITY_pred is None:
            SH_pred = np.argmax(SH_counts)
            VASCULARITY_pred = np.argmax(VASCULARITY_counts)

        SH_preds.append(SH_pred)
        VASCULARITY_preds.append(VASCULARITY_pred)
        combined_pred = max(SH_pred, VASCULARITY_pred)
        combined_preds.append(combined_pred)
            
    return SH_preds, VASCULARITY_preds, combined_preds

if __name__ == '__main__':
    GLOBAL_DICT['train_repeat'] = 2
    with open('./thresh_dict', 'rb') as f:
        d = pickle.load(f)
    GLOBAL_DICT['GSDOPPLER_SH_THRESH_DICT'] = d['GSDOPPLER_SH_THRESH_DICT']
    GLOBAL_DICT['DOPPLER_VASCULARITY_THRESH_DICT'] = d['DOPPLER_VASCULARITY_THRESH_DICT']
    
    # 5 models separately predict using ECOC
    all_SH_preds = single_model_prediction('GSDOPPLER_SH')
    all_VASCULARITY_preds = single_model_prediction('DOPPLER_VASCULARITY')

    # multi-task multi-model ensemble (MULTITUDE)
    SH_preds, VASCULARITY_preds, combined_preds = do_MULTITUDE(all_SH_preds, all_VASCULARITY_preds)

    # save predictions
    predictions = {
            'SH': np.array(SH_preds).astype(np.uint8).tolist(),
            'VASCULARITY': np.array(VASCULARITY_preds).astype(np.uint8).tolist(),
            'combined': np.array(combined_preds).astype(np.uint8).tolist(),
        }
    with open(os.path.join(save_dir, 'predictions.json'), 'w') as f:
        json.dump(predictions, f)

    # evaluate accuracy and linearly weighted kappa
    with open(test_dataset, 'r') as f:
        dataset = json.load(f)
    ground_truth = dict()
    ground_truth['SH'] = np.array([sample['SH_label'] for sample in dataset])
    ground_truth['VASCULARITY'] = np.array([sample['vascularity_label'] for sample in dataset])
    ground_truth['combined'] = np.array([max(sample['SH_label'], sample['vascularity_label']) for sample in dataset])
    for task in ['SH', 'VASCULARITY', 'combined']:
        print(task, ':')
        acc, acc_low, acc_upp = statistic_util.bootstrap_CI(predictions[task], ground_truth[task], statistic_util.accuracy)
        print('Accuracy: {:.1f}({:.1f}-{:.1f})'.format(acc*100, acc_low*100, acc_upp*100))
        kappa, kappa_low, kappa_upp = statistic_util.bootstrap_CI(predictions[task], ground_truth[task], statistic_util.linearly_weighted_kappa)
        print('Linearly weighted kappa: {:.3f}({:.3f}-{:.3f})'.format(kappa, kappa_low, kappa_upp))