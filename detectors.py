import os
import pickle
import numpy as np
import multiprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import scipy as sp
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN

#from methods.autood.autood_aug import run_autood_aug

def set_detector_params(name, X):
    '''
    Method that prepares the unsupervised detectors' hyper-parameters

    Parameters 
    ----------
    name: The name of our dataset
    X: input dataset 

    Returns
    -------
    lof_krange: LOF parameter range
    N_range: Number of outliers range
    knn_krange: KNN parameter range
    if_range: IF parameter range
    mahalanobis_N_range: Mahalanobis parameter range
    '''
    if name == 'spambase':
        K = 9
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        # mahalanobis_N_range=[N]
        mahalanobis_N_range = [1400, 1500, 1600, 1700, 1800, 1900]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'pageblocks':
        K = 80
        N = 560
        class_balance = [0.9, 0.1]
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [300, 400, 500, 600, 700, 800]
        N_size = 6
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'pima':
        K = 100
        #N = 268
        #num_outliers = [N, N, N, N]
        #class_balance = [1 - N / 768.0, N / 768.0]
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        #mahalanobis_N_range = [N]
        mahalanobis_N_range = [220, 230, 240, 250, 260, 270]

        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'shuttle':
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [1000, 1500, 2000, 2500, 3000, 3500]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'http':
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [5000, 10000, 15000, 20000, 25000, 30000]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'annthyroid':
        N = 534
        num_outliers = [N, N, N, N]
        class_balance = [1 - N / 7129.0, N / 7129.0]
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        # mahalanobis_N_range=[N]
        mahalanobis_N_range = [300, 400, 500, 600, 700, 800]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'musk':
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        #mahalanobis_N_range = [100, 120, 140, 160, 180, 200]
        mahalanobis_N_range = [50, 100, 150,200, 250, 300]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'satimage':
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [60, 80, 100, 120, 140, 160]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'pendigits':
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [150, 200, 250, 300, 350, 400]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'mammography':
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [150, 300, 500, 600, 900, 1000]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'satellite':
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [150, 300, 500, 600, 900, 1000]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'cover':
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [150, 300, 500, 600, 900, 1000]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'kddcup99':
        N = 200
        num_outliers = [N, N, N, N]
        class_balance = [1 - N / 48113.0, N / 48113.0]
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [500, 1000, 1500, 2000, 2500, 3000]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'aloi':
        N = 200
        num_outliers = [N, N, N, N]
        class_balance = [1 - N / 48113.0, N / 48113.0]
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [500, 1000, 1500, 2000, 2500, 3000]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif "Friday" in name or "Thursday" in name:
        lof_krange = list(range(10, 100, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [5000, 10000, 15000, 20000, 25000, 30000]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'cardio':
        lof_krange = list(range(10, 110, 10)) * 6  # set
        knn_krange = list(range(10, 110, 10)) * 6  # set
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6  # set
        mahalanobis_N_range = [int(np.shape(X)[0] * percent) for percent in [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]]  # need
        N_range = np.sort(mahalanobis_N_range * 10)  # need
        if_N_range = np.sort(mahalanobis_N_range * 5)

    elif name == 'thyroid':
        lof_krange = list(range(10, 110, 10)) * 6  # set
        knn_krange = list(range(10, 110, 10)) * 6  # set
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6  # set
        mahalanobis_N_range = [int(np.shape(X)[0] * percent) for percent in [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]]  # need
        N_range = np.sort(mahalanobis_N_range * 10)  # need
        if_N_range = np.sort(mahalanobis_N_range * 5)

    elif name == 'smtp':
        lof_krange = list(range(10,110,10)) * 6
        knn_krange = list(range(10,110,10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        # mahalanobis_N_range=[20, 40, 60, 80, 100, 120]
        mahalanobis_N_range=[30, 60, 90, 120, 150, 180]
        # mahalanobis_N_range = [30, 40, 50, 60, 70,80]
        # mahalanobis_N_range = [200, 400, 600, 800, 1000, 1200]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range *10)

    else:
        print("ERROR: The requested dataset does not have parameters for AutoOD unsupervised detectors. Please add dataset to set_detector_params.py")

    return lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range, if_N_range


def get_predictions_scores(scores, num_outliers=400, method_name='LOF', y=None):
    '''
    Method that converts the outlier scores to predictions

    Parameters 
    ----------
    scores: outlier scores from detector for each instance
    num_outliers: number of outliers

    Returns
    -------
    predictions: predictions 0 for inlier 1 for outlier
    scores: same as input
    '''    
    threshold = np.sort(scores)[::-1][num_outliers]
    # threshold, max_f1 = get_best_f1_score(y, lof_scores)
    predictions = np.array(scores > threshold)
    predictions = np.array([int(i) for i in predictions])
    #     print('F1 for {} : {}'.format(method_name, metrics.f1_score(y, predictions)))
    return predictions, scores, f1_score(y, predictions)


############################## Run Detectors ##############################
def run_lof(X, y, k=60):
    clf = LocalOutlierFactor(n_neighbors=k)
    clf.fit(X)
    lof_scores = -clf.negative_outlier_factor_
    return lof_scores

def run_knn(X, y, k=60):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    knn_dists = neigh.kneighbors(X)[0][:, -1]
    return knn_dists

def run_isolation_forest(X, y, max_features=1.0):
    # training the model
    clf = IsolationForest(random_state=42, max_features=max_features)
    clf.fit(X)
    # predictions
    sklearn_score_anomalies = clf.decision_function(X)
    if_scores = [-1 * s + 0.5 for s in sklearn_score_anomalies]
    return if_scores

def mahalanobis(x):
    """Compute the Mahalanobis Distance between each row of x and the data
    """
    x_minus_mu = x - np.mean(x)
    cov = np.cov(x.T)
    inv_covmat = sp.linalg.inv(cov)
    results = []
    x_minus_mu = np.array(x_minus_mu)
    for i in range(np.shape(x)[0]):
        cur_data = x_minus_mu[i, :]
        results.append(np.dot(np.dot(x_minus_mu[i, :], inv_covmat), x_minus_mu[i, :].T))
    return np.array(results)

def run_mahalanobis(X, y):
    # training the model
    dist = mahalanobis(x=X)
    return dist
############################## Run Detectors ##############################


def autood_preprocessing(X, y, lof_krange, knn_krange, if_range, mahalanobis_N_range, if_N_range, N_range):
    '''
    Method that runs each detector with different hyper-parameter configurations

    Parameters 
    ----------
    X: Training data
    lof_krange: range of parameters for lof
    knn_krange: range of parameters for knn
    if_range: range of parameters for if
    mahalanobis_N_range: range of parameters for mahalanobis
    if_N_range: range for number of outliers for if
    N_range: range for number of outliers

    Returns
    -------
    L: predictions for each instance from each detector
    scores: outlier score for each instance from each detector
    
    '''
    all_results = []
    all_scores = []
    f1s = []
    
    method_to_bestf1 = {}
    best_f1 = 0

    temp_lof_results = dict()
    unique_lof_ks = list(set(lof_krange))

    best_lof_f1 = 0
    best_lof_precision = 0
    best_lof_recall = 0
    for k in unique_lof_ks:
        lof_scores = run_lof(X, y, k=k)
        temp_lof_results[k] = lof_scores
    for i in range(len(lof_krange)):
        lof_predictions, lof_scores, f1 = get_predictions_scores(temp_lof_results[lof_krange[i]], num_outliers=N_range[i], method_name='LOF', y=y)
        all_results.append(lof_predictions)
        all_scores.append(lof_scores)
        f1s.append(f1)
        if f1 > best_f1:
            best_f1 = f1

    best_lof_f1 = 0
    for i in np.sort(unique_lof_ks):
        temp_f1 = max(np.array(f1s[0:60])[np.where(np.array(lof_krange) == i)[0]])
        best_lof_f1 = max(best_lof_f1, temp_f1)

    print(f"best lof f1: {best_lof_f1}")

    method_to_bestf1["LOF"] = best_lof_f1

    temp_knn_results = dict()
    unique_knn_ks = list(set(knn_krange)) 
    for k in unique_knn_ks:
        knn_scores = run_knn(X, y, k=k)
        temp_knn_results[k] = knn_scores
    for i in range(len(knn_krange)):
        knn_predictions, knn_scores,f1 = get_predictions_scores(temp_knn_results[knn_krange[i]], num_outliers=N_range[i], method_name='KNN', y=y)
        all_results.append(knn_predictions)
        all_scores.append(knn_scores)
        f1s.append(f1)
        if f1 > best_f1:
            best_f1 = f1
    best_knn_f1 = 0
    for i in np.sort(unique_knn_ks):
        temp_f1 = max(np.array(f1s[60:120])[np.where(np.array(knn_krange) == i)[0]])
        best_knn_f1 = max(best_knn_f1, temp_f1)
    method_to_bestf1["KNN"] = best_knn_f1

    print(f"best KNN f1: {best_knn_f1}")

    temp_if_results = dict()
    unique_if_features = list(set(if_range)) 
    for k in unique_if_features:
        if_scores = run_isolation_forest(X, y, max_features=k)
        temp_if_results[k] = if_scores
    for i in range(len(if_range)):
        if_predictions, if_scores,f1 = get_predictions_scores(temp_if_results[if_range[i]], num_outliers=if_N_range[i], method_name='IF', y=y)
        all_results.append(if_predictions)
        all_scores.append(if_scores)
        f1s.append(f1)
        if f1 > best_f1:
            best_f1 = f1
    best_if_f1 = 0
    for i in np.sort(unique_if_features):
        temp_f1 = max(np.array(f1s[120:150])[np.where(np.array(if_range) == i)[0]])
        best_if_f1 = max(best_if_f1, temp_f1)
    method_to_bestf1["IF"] = best_if_f1

    print(f"best lof IF: {best_if_f1}")

    mahalanobis_scores = run_mahalanobis(X, y)
    best_mahala_f1 = 0
    for i in range(len(mahalanobis_N_range)):
        mahalanobis_predictions,mahalanobis_scores,f1 = get_predictions_scores(mahalanobis_scores, num_outliers=mahalanobis_N_range[i], method_name='mahala', y=y)
        all_results.append(mahalanobis_predictions)
        all_scores.append(mahalanobis_scores)
        best_mahala_f1 = max(best_mahala_f1, f1)
        f1s.append(f1)
        if f1 > best_f1:
            best_f1 = f1
    method_to_bestf1["Mahala"] = best_mahala_f1
    best_method = ""
    best_f1 =0
    for method, f1 in method_to_bestf1.items():
        if f1 > best_f1:
            best_method = method
            best_f1 = f1

    print(f"best mahal f1: {best_mahala_f1}")

    print(f"Best Method = {best_method}, Best F1 = {best_f1}")
    L = np.stack(all_results).T
    scores = np.stack(all_scores).T
    print(f"Best F1 = {best_f1}")


    # with multiprocessing.Pool(8) as pool:
    #     temp_lof_results = dict()
    #     unique_lof_ks = list(set(lof_krange))

    #     processes = [pool.apply_async(run_lof, args=(X, y, k)) for k in lof_krange]
    #     for p in processes:
    #         k, lof_scores = p.get()
    #         temp_lof_results[k] = lof_scores

    #     for i in range(len(lof_krange)):
    #         lof_predictions, lof_scores, f1 = get_predictions_scores(temp_lof_results[lof_krange[i]], num_outliers=N_range[i], method_name='LOF', y=y)
    #         all_results.append(lof_predictions)
    #         all_scores.append(lof_scores)
    #     best_lof_f1 = 0
    #     for arr in all_results[0:60]:
    #         if f1_score(y, arr) > best_lof_f1:
    #             best_lof_f1 = f1_score(y, arr)
    #     print(f"best lof f1: {best_lof_f1}")

    #     temp_knn_results = dict()
    #     unique_knn_ks = list(set(knn_krange))

    #     processes = [pool.apply_async(run_knn, args=(X, y, k,)) for k in knn_krange]
    #     for p in processes:
    #         k, knn_scores = p.get()
    #         temp_knn_results[k] = knn_scores

    #     for i in range(len(knn_krange)):
    #         knn_predictions, knn_scores, f1 = get_predictions_scores(temp_knn_results[knn_krange[i]], num_outliers=N_range[i], method_name='KNN', y=y)
    #         all_results.append(knn_predictions)
    #         all_scores.append(knn_scores)

    #     best_knn_f1 = 0
    #     for arr in all_results[60:120]:
    #         if f1_score(y, arr) > best_knn_f1:
    #             best_knn_f1 = f1_score(y, arr)
    #     print(f"best KNN f1: {best_knn_f1}")

    #     temp_if_results = dict()
    #     unique_if_features = list(set(if_range))

    #     processes = [pool.apply_async(run_isolation_forest, args=(X, y, k,)) for k in unique_if_features]
    #     for p in processes:
    #         k, if_scores = p.get()
    #         temp_if_results[k] = if_scores

    #     for i in range(len(if_range)):
    #         if_predictions, if_scores, f1 = get_predictions_scores(temp_if_results[if_range[i]], num_outliers=if_N_range[i], method_name='IF', y=y)
    #         all_results.append(if_predictions)
    #         all_scores.append(if_scores)

    #     best_if_f1 = 0
    #     for arr in all_results[120:150]:
    #         if f1_score(y, arr) > best_if_f1:
    #             best_if_f1 = f1_score(y, arr)
    #     print(f"best IF f1: {best_if_f1}")

    #     mahalanobis_scores = run_mahalanobis(X, y)
    #     for i in range(len(mahalanobis_N_range)):
    #         mahalanobis_predictions, mahalanobis_scores, f1 = get_predictions_scores(mahalanobis_scores, num_outliers=mahalanobis_N_range[i], method_name='mahala', y=y)
    #         all_results.append(mahalanobis_predictions)
    #         all_scores.append(mahalanobis_scores)

    #     best_mahal_f1 = 0
    #     for arr in all_results[150:156]:
    #         if f1_score(y, arr) > best_mahal_f1:
    #             best_mahal_f1 = f1_score(y, arr)
    #     print(f"best mahal f1: {best_mahal_f1}")
        
    #     L = np.stack(all_results).T
    #     scores = np.stack(all_scores).T
    return L, scores


def set_detector_params_robust(dataset, X_train, outlier_percentage_min, outlier_percentage_max):
    k_range_def = list(range(10, 110, 10))
    if_range_def = [0.5, 0.6, 0.7, 0.8, 0.9]
    N_range_def = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    print(f"Outlier Range defined as [{outlier_percentage_min}%, {outlier_percentage_max}%]")
    outlier_percentage_min = outlier_percentage_min * 0.01
    outlier_percentage_max = outlier_percentage_max * 0.01
    interval = (outlier_percentage_max - outlier_percentage_min) / 5
    
    N_range = [round(x, 5) for x in np.arange(outlier_percentage_min, outlier_percentage_max + interval, interval)]
    N_range = [int(np.shape(X_train)[0] * percent) for percent in N_range]
    knn_N_range = np.sort(N_range * len(k_range_def))
    if_range = [int(np.shape(X_train)[0] * percent) for percent in N_range]


def run_autood_ensemble(args, X_train, y_train):
    if os.path.exists(f"./pickle/{args.dataset}_{args.seed}_{args.train_split}_split_training.pickle"):  # if we have results already saved 
        print("Reading in existing results from unsupervised detectors")
        dataset_results = pickle.load(open(f"./pickle/{args.dataset}_{args.seed}_{args.train_split}_split_training.pickle", "rb"))
        autoOD_L = dataset_results['L']
        scores = dataset_results['scores']
    
    # else run detectors to get their results
    else:
        print("No existing results from unsupervised detectors found for training set. Running unsupervised detectors...")
        
        # get autoOD unsup detect results for train set
        lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range, if_N_range = set_detector_params(args.dataset, X_train)  # set hard coded detect params (optimal values from autoOD)

        # adjust number of outliers for test train split
        mahalanobis_N_range = list((np.array(mahalanobis_N_range) * args.train_split).astype(int))
        if_N_range = list((np.array(if_N_range) * args.train_split).astype(int))
        N_range = list((np.array(N_range) * args.train_split).astype(int))

        autoOD_L, scores = autood_preprocessing(X_train, y_train, lof_krange, knn_krange, if_range, mahalanobis_N_range, if_N_range, N_range)  # run detectors

        # save the results to pickle so we do not have to rerun each time
        dataset_results = {'L': autoOD_L, 'scores': scores}
        os.makedirs("./pickle", exist_ok=True)
        pickle.dump(dataset_results, open(f"./pickle/{args.dataset}_{args.seed}_{args.train_split}_split_training.pickle", "wb"))
    
    # majority vote from unsup detectors for train data
    detector_mv = np.zeros((*autoOD_L.shape, 2))
    for i in range(autoOD_L.shape[0]):
        for j in range(autoOD_L.shape[1]):
            detector_mv[i,j,autoOD_L[i,j]] = 1
    
    pred_labels = np.argmax(np.sum(detector_mv, axis=1), axis=1)
    f1 = f1_score(y_train, pred_labels)
    print("F1 score of Majority Vote by all Unsupervised Detectors:", f1)
    print(f"Accuracy of Majority Vote by all Unsupervised Detectors: {accuracy_score(y_train, pred_labels)}")

    return scores, autoOD_L, pred_labels


def run_if_(args, X, y, outlier_ratio):
    if_model = IForest(contamination=outlier_ratio)
    if_model.fit(X)
    pred_labels = if_model.predict(X)
    outlierness_score = if_model.decision_function(X)
    print(f"F1 score of Isolation Forest: {f1_score(y, pred_labels)}")
    print(f"Accuracy of Isolation Forest: {accuracy_score(y, pred_labels)}")
    print(f"ROCAUC score of Isolation Forest: {roc_auc_score(y, outlierness_score)}")
    
    # get noise rate for the outlier class for co-teaching methods
    gt_outliers = np.where(y == 1)[0]
    noise_rate_outlier = 1-accuracy_score(y[gt_outliers], pred_labels[gt_outliers])
    gt_inliers = np.where(y == 0)[0]
    noise_rate_inlier = 1-accuracy_score(y[gt_inliers], pred_labels[gt_inliers])
    return outlierness_score, 0, pred_labels, noise_rate_outlier, noise_rate_inlier


def run_KNN_(args, X, y):
    knn_model = KNN()
    knn_model.fit(X)
    pred_labels = knn_model.predict(X)
    outlierness_score = knn_model.decision_function(X)
    f1 = f1_score(y, pred_labels)
    print("F1 score of KNN:", f1)
    print(f"Accuracy of KNN: {accuracy_score(y, pred_labels)}")
    return outlierness_score, 0, pred_labels

def run_lof_(args, X, y):
    lof_model = LOF()
    lof_model.fit(X)
    pred_labels = lof_model.predict(X)
    outlierness_score = lof_model.decision_function(X)
    f1 = f1_score(y, pred_labels)
    print("F1 score of LOF:", f1)
    print(f"Accuracy of LOF: {accuracy_score(y, pred_labels)}")
    return outlierness_score, 0, pred_labels


def run_detectors(args, X, y, outlier_ratio):
    '''
    Method that generates our initial noisy pseudo labels

    Parameters 
    ----------
    args: arguments provided by user
    X: data
    y: GT labels -> only used to get f1 score of detectors
    outlier_ratio: ratio of outliers in dataset
        
    Returns
    -------
    pred_labels: initial noisy pseudo labels y_noisy 
    outlier_score: outlier score for each instance
    autoOD_L: Only for ensemble, each detectors prediction for each instance
    '''
    if args.method == "augment":
        # autoOD augment requires multiple labels for each sample: required ensemble to be ran
        outlier_score, autoOD_L, pred_labels = run_autood_ensemble(args, X, y)
    else:
        if args.label_noise == "ensemble_mv":
            # run the ensemble of unsupervised OD methods from AutoOD, use majority vote as noisy labels
            outlier_score, autoOD_L, pred_labels = run_autood_ensemble(args, X, y)
        
        if args.label_noise == "isolation_forest":
            # run only isolation forest to get noisy labels
            outlier_score, autoOD_L, pred_labels, noise_rate_outlier, noise_rate_inlier = run_if_(args, X, y, outlier_ratio)
        
        if args.label_noise == "knn":
            # run only knn to get noisy labels
            outlier_score, autoOD_L, pred_labels = run_KNN_(args, X, y)   

        if args.label_noise == "lof":
            # run only lof to get noisy labels
            outlier_score, autoOD_L, pred_labels = run_lof_(args, X, y)

        if args.label_noise == "gt":
            # run with ground truth labels
            autoOD_L = 0
            pred_labels = y
            f1 = f1_score(y, pred_labels)
            print("F1 score of GT labels is (should be 1):", f1)
    

    return pred_labels, outlier_score, autoOD_L, noise_rate_outlier, noise_rate_inlier

