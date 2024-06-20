import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
import sys
from sklearn import metrics


def run_autood_augment(args, train_data, test_data, X_train, y_train, X_test, y_test, dim, detector_mv_train, detector_mv_test, scores_train, autoOD_L_2d):
    print("Running autoOD augment")
    L = autoOD_L_2d
    X = X_train
    y = y_train
    scores = scores_train
    max_iteration = 50

    # prepare scores for training
    L_prev = L
    prediction_result_list = []
    classifier_result_list = []
    prediction_list = []
    cur_f1_scores = []
    prediction_high_conf_outliers = np.array([])
    prediction_high_conf_inliers = np.array([])
    prediction_classifier_disagree = np.array([])
    index_range = np.array([[0, 60], [60, 120], [120, 150], [150, 156]])
    coef_index_range = np.array([[0, 10], [10, 20], [20, 25], [25, 26]])
    coef_remain_index = range(156)
    scores_for_training_indexes = []
    for i in range(len(index_range)):
        start = index_range[i][0]
        temp_range = coef_index_range[i][1] - coef_index_range[i][0]
        scores_for_training_indexes = scores_for_training_indexes + list(range(start, start + temp_range))
    scores_for_training = scores[:, np.array(scores_for_training_indexes)]

    # first iteration
    high_confidence_threshold = 0.99
    low_confidence_threshold = 0.01
    max_iter = 200
    remain_params_tracking = np.array(range(0, np.max(coef_index_range)))
    training_data_F1 = []
    two_prediction_corr = []

    min_max_diff = []
    N_size = 6

    last_training_data_indexes = []
    counter = 0

    print("##################################################################")
    print("Start First-round AutoOD training...")
    for i_range in range(0, max_iteration):
        num_methods = np.shape(L)[1]  # L is matrix: data points x unsup methods, L[0] gives us the number of detectors
        print(f"Number of detectors: {len(L[0])}")
        agree_outlier_indexes = np.sum(L, axis=1) == np.shape(L)[1]  # index of all data points where they all agree on outlier
        agree_inlier_indexes = np.sum(L, axis=1) == 0
        print(f"Total number of data points where all detectors agree on pred: {np.sum(np.sum(L, axis=1) == np.shape(L)[1])}")
        disagree_indexes = np.where(np.logical_or(np.sum(L, axis=1) == 0, np.sum(L, axis=1) == num_methods) == 0)[0]
        all_inlier_indexes = np.setdiff1d(np.where(agree_inlier_indexes)[0], prediction_high_conf_outliers)
        if len(prediction_high_conf_inliers) > 0:
            all_inlier_indexes = np.intersect1d(
                np.setdiff1d(np.where(agree_inlier_indexes)[0], prediction_high_conf_outliers),
                prediction_high_conf_inliers)
        all_outlier_indexes = np.union1d(np.where(agree_outlier_indexes)[0], prediction_high_conf_outliers)  # set with all outliers that agree
        all_inlier_indexes = np.setdiff1d(all_inlier_indexes, prediction_classifier_disagree)  #  set with all inliers that agree        

        self_agree_index_list = []
        if ((len(all_outlier_indexes) == 0) or (len(all_inlier_indexes) / len(all_outlier_indexes) > 1000)):  # if no outliers in reliable set: loosen requirements
            for i in range(0, len(index_range)):
                if (index_range[i, 1] - index_range[i, 0] <= 6):
                    continue
                temp_index = disagree_indexes[np.where(np.sum(L[disagree_indexes][:, index_range[i, 0]: index_range[i, 1]], axis=1) == (index_range[i, 1] - index_range[i, 0]))[0]]
                self_agree_index_list = np.union1d(self_agree_index_list, temp_index)
            self_agree_index_list = [int(i) for i in self_agree_index_list]

        all_outlier_indexes = np.union1d(all_outlier_indexes, self_agree_index_list)  # add empty set if outliers are in reliable set, else add self agree as well
        if(len(np.setdiff1d(all_outlier_indexes, prediction_classifier_disagree)) != 0):  # DH: remove reliable only if set will not be 0
            all_outlier_indexes = np.setdiff1d(all_outlier_indexes, prediction_classifier_disagree) 

        #### DH
        self_agree_index_list_inlier = []
        if ((len(all_inlier_indexes) == 0)):  # DH: requierments are relaxed 
            for i in range(0, len(index_range)):
                if (index_range[i, 1] - index_range[i, 0] <= 6):
                    continue
                temp_index = disagree_indexes[np.where(np.sum(L[disagree_indexes][:, index_range[i, 0]: index_range[i, 1]], axis=1) == (index_range[i, 1] - index_range[i, 0]))[0]]
                self_agree_index_list_inlier = np.union1d(self_agree_index_list_inlier, temp_index)
            self_agree_index_list_inlier = [int(i) for i in self_agree_index_list_inlier]
            
        all_inlier_indexes = np.union1d(all_inlier_indexes, self_agree_index_list_inlier)  
        if(len(np.setdiff1d(all_inlier_indexes, prediction_classifier_disagree)) != 0): 
                all_inlier_indexes = np.setdiff1d(all_inlier_indexes, prediction_classifier_disagree)  

        if(len(all_inlier_indexes) == 0):  # still no reliable inliers: add most reliable inlier
            min_score = sys.maxsize
            index_method = 0
            counter = -1
            for scores_method in scores:
                counter = counter + 1
                if np.average(scores_method) < min_score:
                    min_score = np.average(scores_method)
                    index_method = counter
            all_inlier_indexes = np.union1d(all_inlier_indexes, index_method) 
        #### DH

        data_indexes = np.concatenate((all_inlier_indexes, all_outlier_indexes), axis=0)  # all reliable indices
        data_indexes = np.array([int(i) for i in data_indexes])
        labels = np.concatenate((np.zeros(len(all_inlier_indexes)), np.ones(len(all_outlier_indexes))), axis=0)
        transformer = RobustScaler().fit(scores_for_training)
        scores_transformed = transformer.transform(scores_for_training)
        training_data = scores_transformed[data_indexes]
        if y is not None:
            training_data_F1.append(metrics.f1_score(y[data_indexes], labels))  # y is ground truth, labels are predicted
        
        clf = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(training_data, labels)
        clf_predictions = clf.predict(scores_transformed)
        clf_predict_proba = clf.predict_proba(scores_transformed)[:, 1]

        transformer = RobustScaler().fit(X)
        X_transformed = transformer.transform(X)
        X_training_data = X_transformed[data_indexes]

        clf_X = SVC(gamma='auto', probability=True, random_state=0)
        clf_X.fit(X_training_data, labels)
        clf_predict_proba_X = clf_X.predict_proba(X_transformed)[:, 1]
        if y is not None:
            print("#######################################################")
            print(f'Iteration = {i_range}, F-1 score = {metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > 0.5]))}')
        else:
            print(f"Iteration = {i_range}")
        prediction_result_list.append(clf_predict_proba)
        classifier_result_list.append(clf_predict_proba_X)

        prediction_list.append(np.array([int(i) for i in clf_predictions]))

        prediction_high_conf_outliers = np.intersect1d(  # DH select high conf outliers
            np.where(prediction_result_list[-1] > high_confidence_threshold)[0],  # results from logistic regression
            np.where(classifier_result_list[-1] > high_confidence_threshold)[0])  # results from SVM

        prediction_high_conf_inliers = np.intersect1d(  # DH select high conf inliers
            np.where(prediction_result_list[-1] < low_confidence_threshold)[0],
            np.where(classifier_result_list[-1] < low_confidence_threshold)[0])

        temp_prediction = np.array([int(i) for i in prediction_result_list[-1] > 0.5])  # logistic
        temp_classifier = np.array([int(i) for i in classifier_result_list[-1] > 0.5])  # SVM
        prediction_classifier_disagree = np.where(temp_prediction != temp_classifier)[0]  # gets index of where do SVM and log. disagree

        two_prediction_corr.append(np.corrcoef(clf_predict_proba, clf_predict_proba_X)[0, 1])  # Return Pearson product-moment correlation coefficients

        if np.max(coef_index_range) >= 2:
            if (len(prediction_high_conf_outliers) > 0 and len(prediction_high_conf_inliers) > 0):
                # reliabel object set, SVM and log. agree
                new_data_indexes = np.concatenate((prediction_high_conf_outliers, prediction_high_conf_inliers), axis=0)
                new_data_indexes = np.array([int(i) for i in new_data_indexes])
                # update new labels based on SVM and log.
                new_labels = np.concatenate((np.ones(len(prediction_high_conf_outliers)), np.zeros(len(prediction_high_conf_inliers))), axis=0)
                clf_prune_2 = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(scores_transformed[new_data_indexes], new_labels)  # train log with updated labels: this is to prune poor detectors
                combined_coef = clf_prune_2.coef_[0]
            else:
                combined_coef = clf.coef_[0]

            if(np.max(coef_index_range) >= 2):
                if (len(set(combined_coef)) > 1):
                    cur_clf_coef = combined_coef
                    cutoff = max(max(0, np.mean(combined_coef) - np.std(combined_coef)), min(combined_coef))  # cutoff to purne poor detectors

                    remain_indexes_after_cond = (
                            cur_clf_coef > cutoff)  # np.logical_and(cur_clf_coef > cutoff, abs(cur_clf_coef) > 0.01) 
                    remain_params_tracking = remain_params_tracking[remain_indexes_after_cond]  # prune poor detectors

                    remain_indexes_after_cond_expanded = []  # update index for reliable set: Can we use this to keep track of when detectors are removed?
                    for i in range(0, len(coef_index_range)):  
                        s_e_range = coef_index_range[i, 1] - coef_index_range[i, 0]
                        s1, e1 = coef_index_range[i, 0], coef_index_range[i, 1]
                        s2, e2 = index_range[i, 0], index_range[i, 1]
                        saved_indexes = np.where(cur_clf_coef[s1:e1] > cutoff)[0]
                        for j in range(N_size):
                            remain_indexes_after_cond_expanded.extend(np.array(saved_indexes) + j * s_e_range + s2)

                    new_coef_index_range_seq = []
                    for i in range(0, len(coef_index_range)): 
                        s, e = coef_index_range[i, 0], coef_index_range[i, 1]
                        new_coef_index_range_seq.append(sum((remain_indexes_after_cond)[s:e]))
                    
                    prev_index_range = np.copy(index_range)
                    prev_coef_index_range = np.copy(coef_index_range)
                    coef_index_range = []
                    index_range = []
                    cur_sum = 0
                    temp = 0
                    for i in range(0, len(new_coef_index_range_seq)):
                        coef_index_range.append([cur_sum, cur_sum + new_coef_index_range_seq[i]])
                        # 6 should be changed to something like total detectors/specif. e.g. 60/10 or 33/10
                        index_range.append([cur_sum * 6, 6 * (cur_sum + new_coef_index_range_seq[i])])  
                        cur_sum += new_coef_index_range_seq[i]

                    coef_index_range = np.array(coef_index_range)
                    index_range = np.array(index_range)

                    L = L[:, remain_indexes_after_cond_expanded]  # DH: here we are updating detectors
                    scores_for_training = scores_for_training[:, remain_indexes_after_cond]

        if ((len(last_training_data_indexes) == len(data_indexes)) and
                (sum(last_training_data_indexes == data_indexes) == len(data_indexes)) and
                (np.max(coef_index_range) < 2)):
            counter = counter + 1  # early stop statment: no changes for 3 iterations -> stop

        else:
            counter = 0
        if (counter > 3):
            break
        last_training_data_indexes = data_indexes

    # second round
    scores_for_training_indexes = []
    for i in range(len(index_range)):
        start = index_range[i][0]
        temp_range = coef_index_range[i][1] - coef_index_range[i][0]
        scores_for_training_indexes = scores_for_training_indexes + list(range(start, start + temp_range))
    
    scores_for_training = scores[:, np.array(scores_for_training_indexes)]
    transformer = RobustScaler().fit(scores_for_training)
    scores_transformed = transformer.transform(scores_for_training)
    training_data = scores_transformed[data_indexes]

    if y is not None:
        training_data_F1.append(metrics.f1_score(y[data_indexes], labels))
    
    clf = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(training_data, labels)
    clf_predictions = clf.predict(scores_transformed)
    clf_predict_proba = clf.predict_proba(scores_transformed)[:, 1]

    transformer = RobustScaler().fit(X)
    X_transformed = transformer.transform(X)
    X_training_data = X_transformed[data_indexes]

    clf_X = SVC(gamma='auto', probability=True, random_state=0)
    clf_X.fit(X_training_data, labels)
    clf_predict_proba_X = clf_X.predict_proba(X_transformed)[:, 1]
    if y is not None:
        cur_f1_scores.append(metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > 0.5])))

    prediction_result_list.append(clf_predict_proba)
    classifier_result_list.append(clf_predict_proba_X)

    prediction_list.append(np.array([int(i) for i in clf_predictions]))

    prediction_high_conf_outliers = np.intersect1d(
        np.where(prediction_result_list[-1] > high_confidence_threshold)[0],
        np.where(classifier_result_list[-1] > high_confidence_threshold)[0])
    prediction_high_conf_inliers = np.intersect1d(
        np.where(prediction_result_list[-1] < low_confidence_threshold)[0],
        np.where(classifier_result_list[-1] < low_confidence_threshold)[0])

    temp_prediction = np.array([int(i) for i in prediction_result_list[-1] > 0.5])
    temp_classifier = np.array([int(i) for i in classifier_result_list[-1] > 0.5])
    prediction_classifier_disagree = np.where(temp_prediction != temp_classifier)[0]

    L = L_prev
    remain_params_tracking = np.array(range(0, np.max(coef_index_range)))

    if np.max(coef_index_range) >= 2:
        if (len(prediction_high_conf_outliers) > 0 and len(prediction_high_conf_inliers) > 0):  # losen requirments 
            new_data_indexes = np.concatenate((prediction_high_conf_outliers, prediction_high_conf_inliers), axis=0)
            new_data_indexes = np.array([int(i) for i in new_data_indexes])
            new_labels = np.concatenate(
                (np.ones(len(prediction_high_conf_outliers)), np.zeros(len(prediction_high_conf_inliers))), axis=0)
            clf_prune_2 = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(
                scores_transformed[new_data_indexes], new_labels)
            combined_coef = clf_prune_2.coef_[0]
        else:
            combined_coef = clf.coef_[0]

        if (np.max(coef_index_range) > 2 or
                ((np.max(combined_coef) / np.min(combined_coef) >= 1.1) and np.max(coef_index_range) >= 2)):
            if (len(set(combined_coef)) > 1):
                cur_clf_coef = combined_coef
                cutoff = max(max(0, np.mean(combined_coef) - np.std(combined_coef)), min(combined_coef))

                remain_indexes_after_cond = (cur_clf_coef > cutoff)  # np.logical_and(cur_clf_coef > cutoff, abs(cur_clf_coef) > 0.01) 
                remain_params_tracking = remain_params_tracking[remain_indexes_after_cond]
                remain_indexes_after_cond_expanded = []
                for i in range(0, len(coef_index_range)):  #
                    s_e_range = coef_index_range[i, 1] - coef_index_range[i, 0]
                    s1, e1 = coef_index_range[i, 0], coef_index_range[i, 1]
                    s2, e2 = index_range[i, 0], index_range[i, 1]
                    saved_indexes = np.where(cur_clf_coef[s1:e1] > cutoff)[0]    
                    remain_indexes_after_cond_expanded.extend(np.array(saved_indexes) + j * s_e_range + s2)

                new_coef_index_range_seq = []
                for i in range(0, len(coef_index_range)):  #
                    s, e = coef_index_range[i, 0], coef_index_range[i, 1]
                    new_coef_index_range_seq.append(sum((remain_indexes_after_cond)[s:e]))

                coef_index_range = []
                index_range = []
                cur_sum = 0
                for i in range(0, len(new_coef_index_range_seq)):
                    coef_index_range.append([cur_sum, cur_sum + new_coef_index_range_seq[i]])
                    # 6 should be changed to something like total detectors/specif. e.g. 60/10 or 33/10
                    index_range.append([cur_sum * 6, 6 * (cur_sum + new_coef_index_range_seq[i])]) 
                    cur_sum += new_coef_index_range_seq[i]

                coef_index_range = np.array(coef_index_range)
                index_range = np.array(index_range)
                L = L[:, remain_indexes_after_cond_expanded]
                scores_for_training = scores_for_training[:, remain_indexes_after_cond]

    print("##################################################################")
    print("Start Second-round AutoOD training...")
    last_training_data_indexes = []
    counter = 0
    for i_range in range(0, 50):
        num_methods = np.shape(L)[1]

        agree_outlier_indexes = np.sum(L, axis=1) == np.shape(L)[1]
        agree_inlier_indexes = np.sum(L, axis=1) == 0

        disagree_indexes = np.where(np.logical_or(np.sum(L, axis=1) == 0, np.sum(L, axis=1) == num_methods) == 0)[0]

        all_inlier_indexes = np.setdiff1d(np.where(agree_inlier_indexes)[0], prediction_high_conf_outliers)

        if len(prediction_high_conf_inliers) > 0:
            all_inlier_indexes = np.intersect1d(
                np.setdiff1d(np.where(agree_inlier_indexes)[0], prediction_high_conf_outliers),
                prediction_high_conf_inliers)

        all_outlier_indexes = np.union1d(np.where(agree_outlier_indexes)[0], prediction_high_conf_outliers)
        all_inlier_indexes = np.setdiff1d(all_inlier_indexes, prediction_classifier_disagree)

        self_agree_index_list = []
        if ((len(all_outlier_indexes) == 0) or (len(all_inlier_indexes) / len(all_outlier_indexes) > 1000)):  # loosen requirements 
            for i in range(0, len(index_range)):
                if (index_range[i, 1] - index_range[i, 0] <= 6):
                    continue
                temp_index = disagree_indexes[np.where(
                    np.sum(L[disagree_indexes][:, index_range[i, 0]: index_range[i, 1]], axis=1) == (
                            index_range[i, 1] - index_range[i, 0]))[0]]
                self_agree_index_list = np.union1d(self_agree_index_list, temp_index)
            self_agree_index_list = [int(i) for i in self_agree_index_list]
        
        all_outlier_indexes = np.union1d(all_outlier_indexes, self_agree_index_list)
        if(len(np.setdiff1d(all_outlier_indexes, prediction_classifier_disagree)) != 0):  # remove reliable only if set will not be 0
            all_outlier_indexes = np.setdiff1d(all_outlier_indexes, prediction_classifier_disagree)
        
        #### DH
        self_agree_index_list_inlier = []
        if ((len(all_inlier_indexes) == 0)):  # here is where requierments are relaxed 
            for i in range(0, len(index_range)):
                if (index_range[i, 1] - index_range[i, 0] <= 6):
                    continue
                temp_index = disagree_indexes[np.where(
                    np.sum(L[disagree_indexes][:, index_range[i, 0]: index_range[i, 1]], axis=1) == (
                            index_range[i, 1] - index_range[i, 0]))[0]]
                self_agree_index_list_inlier = np.union1d(self_agree_index_list_inlier, temp_index)
            self_agree_index_list_inlier = [int(i) for i in self_agree_index_list_inlier]
        all_inlier_indexes = np.union1d(all_inlier_indexes, self_agree_index_list_inlier)  
        if(len(np.setdiff1d(all_inlier_indexes, prediction_classifier_disagree)) != 0): 
            all_inlier_indexes = np.setdiff1d(all_inlier_indexes, prediction_classifier_disagree)

        if(len(all_inlier_indexes) == 0):  # still no reliable labels: add most reliable inlier
            min_score = sys.maxsize
            index_method = 0
            counter = -1
            for scores_method in scores:
                counter = counter + 1
                if np.average(scores_method) < min_score:
                    min_score = np.average(scores_method)
                    index_method = counter
            all_inlier_indexes = np.union1d(all_inlier_indexes, index_method) 
        #### DH

        data_indexes = np.concatenate((all_inlier_indexes, all_outlier_indexes), axis=0)
        data_indexes = np.array([int(i) for i in data_indexes])
        labels = np.concatenate((np.zeros(len(all_inlier_indexes)), np.ones(len(all_outlier_indexes))), axis=0)
        transformer = RobustScaler().fit(scores_for_training)
        scores_transformed = transformer.transform(scores_for_training)
        training_data = scores_transformed[data_indexes]
        if y is not None:
            training_data_F1.append(metrics.f1_score(y[data_indexes], labels))

        clf = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(training_data, labels)
        clf_predictions = clf.predict(scores_transformed)
        clf_predict_proba = clf.predict_proba(scores_transformed)[:, 1]

        transformer = RobustScaler().fit(X)
        X_transformed = transformer.transform(X)
        X_training_data = X_transformed[data_indexes]

        clf_X = SVC(gamma='auto', probability=True, random_state=0)
        clf_X.fit(X_training_data, labels)

        clf_predict_proba_X = clf_X.predict_proba(X_transformed)[:, 1]
        if y is not None:
            print("#######################################################")
            print(f'Iteration = {i_range}, F-1 score = {metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > 0.5]))}')
            cur_f1_scores.append(metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > 0.5])))
            #f1_at_iter(f"2_{i_range}", len(X_training_data), len(L[0]), len(all_inlier_indexes), len(all_outlier_indexes), metrics.f1_score(y, np.array([int(i) for i in clf_predict_proba_X > 0.5])))  # DH
        else:
            print(f"Iteration = {i_range}")

        prediction_result_list.append(clf_predict_proba)
        classifier_result_list.append(clf_predict_proba_X)

        prediction_list.append(np.array([int(i) for i in clf_predictions]))

        prediction_high_conf_outliers = np.intersect1d(
            np.where(prediction_result_list[-1] > high_confidence_threshold)[0],
            np.where(classifier_result_list[-1] > high_confidence_threshold)[0])

        prediction_high_conf_inliers = np.intersect1d(
            np.where(prediction_result_list[-1] < low_confidence_threshold)[0],
            np.where(classifier_result_list[-1] < low_confidence_threshold)[0])

        temp_prediction = np.array([int(i) for i in prediction_result_list[-1] > 0.5])
        temp_classifier = np.array([int(i) for i in classifier_result_list[-1] > 0.5])
        prediction_classifier_disagree = np.where(temp_prediction != temp_classifier)[0]

        if np.max(coef_index_range) >= 2:
            if (len(prediction_high_conf_outliers) > 0 and len(prediction_high_conf_inliers) > 0):
                new_data_indexes = np.concatenate((prediction_high_conf_outliers, prediction_high_conf_inliers),
                                                    axis=0)
                new_data_indexes = np.array([int(i) for i in new_data_indexes])
                new_labels = np.concatenate(
                    (np.ones(len(prediction_high_conf_outliers)), np.zeros(len(prediction_high_conf_inliers))),
                    axis=0)
                clf_prune_2 = LogisticRegression(random_state=0, penalty='l2', max_iter=max_iter).fit(
                    scores_transformed[new_data_indexes], new_labels)
                combined_coef = clf_prune_2.coef_[0]
            else:
                combined_coef = clf.coef_[0]

            if (np.max(coef_index_range) > 2 or
                    ((np.max(combined_coef) / np.min(combined_coef) >= 1.1) and np.max(coef_index_range) >= 2)):
                if (len(set(combined_coef)) > 1):
                    cur_clf_coef = combined_coef
                    cutoff = max(max(0, np.mean(combined_coef) - np.std(combined_coef)), min(combined_coef))

                    remain_indexes_after_cond = (
                            cur_clf_coef > cutoff)  # np.logical_and(cur_clf_coef > cutoff, abs(cur_clf_coef) > 0.01) # #
                    remain_params_tracking = remain_params_tracking[remain_indexes_after_cond]
                    remain_indexes_after_cond_expanded = []
                    for i in range(0, len(coef_index_range)):  #
                        s_e_range = coef_index_range[i, 1] - coef_index_range[i, 0]
                        s1, e1 = coef_index_range[i, 0], coef_index_range[i, 1]
                        s2, e2 = index_range[i, 0], index_range[i, 1]
                        saved_indexes = np.where(cur_clf_coef[s1:e1] > cutoff)[0]
                    
                        for j in range(N_size):
                            remain_indexes_after_cond_expanded.extend(np.array(saved_indexes) + j * s_e_range + s2)

                    new_coef_index_range_seq = []
                    for i in range(0, len(coef_index_range)):  #
                        s, e = coef_index_range[i, 0], coef_index_range[i, 1]
                        new_coef_index_range_seq.append(sum((remain_indexes_after_cond)[s:e]))
                    
                    coef_index_range = []
                    index_range = []
                    cur_sum = 0
                    for i in range(0, len(new_coef_index_range_seq)):
                        coef_index_range.append([cur_sum, cur_sum + new_coef_index_range_seq[i]])
                        # 6 should be changed to something like total detectors/specif. e.g. 60/10 or 33/10
                        index_range.append([cur_sum * 6, 6 * (cur_sum + new_coef_index_range_seq[i])]) 
                        cur_sum += new_coef_index_range_seq[i]

                    coef_index_range = np.array(coef_index_range)
                    index_range = np.array(index_range)
                    L = L[:, remain_indexes_after_cond_expanded]
                    scores_for_training = scores_for_training[:, remain_indexes_after_cond]
        
        if ((len(last_training_data_indexes) == len(data_indexes)) and
                (sum(last_training_data_indexes == data_indexes) == len(data_indexes)) and
                (np.max(coef_index_range) < 2)):
            counter = counter + 1
        else:
            counter = 0
        if (counter > 3):
            break
        last_training_data_indexes = data_indexes
    final_pred = np.array([int(i) for i in clf_predict_proba_X > 0.5])
    print(f"Final AutoOD F-1 Score= {metrics.f1_score(y, final_pred)}")   
    return final_pred      
