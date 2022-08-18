import numpy as np
import random
from sklearn import metrics
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
from statsmodels.stats.inter_rater import cohens_kappa

def cv_CI(arr, percentage=True):
    arr = np.array(arr)
    if percentage:
        arr = arr * 100
    std = np.std(arr, axis=0, ddof=1)
    bias = 1.96 * std / np.sqrt(len(arr))
    mean = np.mean(arr)
    return mean, mean-bias, mean+bias

def bootstrap_CI(x, y, func, n=1000):
    x = np.array(x)
    y = np.array(y)
    N = len(y)
    stats = []
    random.seed(0)
    for _ in range(n):
        indices = np.array(random.choices(list(range(N)), k=N))
        tmp_x = x[indices]
        tmp_y = y[indices]
        stat = func(tmp_x, tmp_y)
        stats.append(stat)
    stats.sort()
    stat = func(x, y)
    return stat, stats[int(0.025 * n)], stats[int(0.975 * n)]

def bootstrap_z_test(x1, x2, y, func, n=1000):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)
    N = len(y)
    stats1 = []
    stats2 = []
    random.seed(0)
    for _ in range(n):
        indices = np.array(random.choices(list(range(N)), k=N))
        tmp_x1 = x1[indices]
        tmp_x2 = x2[indices]
        tmp_y = y[indices]
        stat1 = func(tmp_x1, tmp_y)
        stat2 = func(tmp_x2, tmp_y)
        stats1.append(stat1)
        stats2.append(stat2)

    DescrStatsW1 = DescrStatsW(stats1)
    DescrStatsW2 = DescrStatsW(stats2)
    obj = CompareMeans(DescrStatsW1, DescrStatsW2)
    tstat, pvalue = obj.ztest_ind(usevar="unequal")

    stats1.sort()
    stats2.sort()
    stat1 = func(x1, y)
    stat2 = func(x2, y)

    return stat1, stats1[int(0.025 * n)], stats1[int(0.975 * n)], stat2, stats2[int(0.025 * n)], stats2[int(0.975 * n)], pvalue

def accuracy(x, y):
    return np.sum(x == y) / len(y)

def linearly_weighted_kappa(x, y):
    matrix = np.zeros((4,4), np.uint8)
    for a, b in zip(x, y):
        matrix[b][a] += 1
    k = cohens_kappa(matrix, wt='linear')
    return k['kappa']

def sensitivity(x, y):
    tn, fp, fn, tp = metrics.confusion_matrix(y, x).ravel()
    sen = tp / (tp + fn)
    return sen

def specificity(x, y):
    tn, fp, fn, tp = metrics.confusion_matrix(y, x).ravel()
    spec = tn / (tn + fp)
    return spec

def Youden_index(x, y):
    tn, fp, fn, tp = metrics.confusion_matrix(y, x).ravel()
    return tp / (tp+fn) + tn / (tn+fp) - 1

def f1_score(x, y):
    return metrics.f1_score(y, x)
