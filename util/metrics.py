import cv2
import time
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


class Meter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(Meter):
    def __init__(self):
        Meter.__init__(self)

    def start(self):
        self.start_time = time.time()

    def record(self, n = 1):
        spent_time = time.time() - self.start_time
        self.update(spent_time, n)

def accuracy(pred, target):
    correct = (pred.cpu().long() == target)
    return 100.0 * correct.sum() / len(correct)

def accuracy_from_matrix(c_matrix):
    total = np.sum(c_matrix)
    correct = np.trace(c_matrix)
    return 100.0 * correct / total

def precision(c_matrix, ti):
    pre = c_matrix[ti,ti] / np.sum(c_matrix[:,ti])
    return pre

def recall(c_matrix, ti):
    recall = c_matrix[ti,ti] / np.sum(c_matrix[ti])
    return recall

def f_score(c_matrix, ti):
    pre = c_matrix[ti, ti] / np.sum(c_matrix[:, ti])
    recall = c_matrix[ti, ti] / np.sum(c_matrix[ti])
    score = 2 * pre * recall / (pre + recall)
    return score

def comfusion_matrix(preds, labels, c_num):
    if c_num == 1: # single output
        c_num = 2 
    confuse_m = np.zeros((c_num, c_num))
    for i in range(len(labels)):
        label = int(labels[i])
        pred = int(preds[i]) 
        confuse_m[label,pred] += 1
    return confuse_m

def auc_score(y_true, y_scores):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_scores, list):
        y_scores = np.array(y_scores)
    
    if y_scores.shape[1] == 2:
        # binary classification
        try:
            auc = roc_auc_score(y_true, y_scores[:, 1]) 
        except:
            return 1.0
    else:
        # multi-class AUC in 'macro' mode
        label_mask = np.zeros_like(y_scores, dtype=np.uint8)
        for i, label in enumerate(y_true):
            label_mask[i, label] = 1
        auc = roc_auc_score(label_mask, y_scores)

    return auc
    
