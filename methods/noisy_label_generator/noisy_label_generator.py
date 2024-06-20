from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import os
import csv

def record_noisy_labels(args, y, y_noisy, outlier_score):
    '''
    record the performance of the noisy label generator 

    Parameters
    ----------
    args: arguments
    y: GT
    y_noisy: noisy labels
    outlier_score: outlier score

    Returns
    -------
    Nothing: stores results in csv
    '''
    # calculate f1 score and auc 
    noisy_label_generator_f1 = f1_score(y, y_noisy)
    noisy_label_generator_auc = roc_auc_score(y, outlier_score)
    noisy_label_generator_precision = precision_score(y, y_noisy)
    noisy_label_generator_recall = recall_score(y, y_noisy)
    
    print(f"F1 score of {args.label_noise}: {noisy_label_generator_f1}")
    print(f"AUC of {args.label_noise}: {noisy_label_generator_auc}")

    # log results 
    csv_file_path = f"results/final_results/sota_comparison/results.csv"
    path_exists = False
    if os.path.exists(csv_file_path):
        path_exists = True
    
    with open(csv_file_path, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        if not path_exists:
            # column names
            csv_writer.writerow(["Method", "Dataset", "seed", "batch_norm", "f1", "precision", "recall", "auc"])
        csv_writer.writerow([args.label_noise, args.dataset, args.seed, args.batch_norm, noisy_label_generator_f1, noisy_label_generator_precision, noisy_label_generator_recall, 
                             noisy_label_generator_auc])
