import argparse
import random
import numpy as np
import torch
import wandb
from sklearn.metrics import f1_score

from load_data import load_data, create_dataloader, load_synthetic_data
from detectors import run_detectors
# from methods.id_autood.id_autood import run_id_autood
from methods.autood.autood_aug import run_autood_augment
from methods.autood.autood_clean import run_autood_clean
from methods.uadb.main import run_UADB
from methods.co_teaching.co_teaching import run_coteaching
from methods.co_teaching_plus.co_teaching_plus import run_coteaching_plus
from methods.jocor.jocor import run_jocor
from methods.unity.unity import run_unity
from methods.selfie.selfie import run_selfie
from methods.seal.seal import run_seal
from methods.noisy_label_generator.noisy_label_generator import record_noisy_labels

if __name__ == "__main__":
    # parsing and configuration
    parser = argparse.ArgumentParser(description="Codebase for different unsupervised outlier detection methods")
    parser.add_argument('--lr_co_teaching', type=float, default=0.01,help='learning rate (default: 0.001)')
    parser.add_argument('--lr_contrastive', type=float, default=0.001,help='learning rate (default: 0.001)')
    parser.add_argument('--hidden_dim', type=int, default=128,help='hidden dim for the co-teaching models')
    parser.add_argument('--label_noise', type=str, default="isolation_forest",help='how should noisy labels be generated')
    parser.add_argument('--momentum', type=float, default=0.9,help='momentum for optimizer')
    parser.add_argument('--batch_size', type=int, default=128,help='batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200,help='number of epochs to train (default: 20)')
    parser.add_argument('--col_name_labels', default='label', type=str, help='dataset column name for labels')
    parser.add_argument('--w_b', default=0, type=int, help='w&b')
    parser.add_argument('--strategy', default="split", type=str, help='method')
    parser.add_argument('--dataset', default="annthyroid", type=str, help='dataset')  
    parser.add_argument('--method', default="UNITY", type=str, help='method')
    parser.add_argument('--seed', default=9, type=int,help='random seed')
    parser.add_argument('--epochs_warmup', default=15, type=int, help='number of epochs to run warmup for (no sample selection)')
    parser.add_argument('--beta', default=0, type=float, help='controls how much weight should be given to historical data vs newer data for EMA')
    parser.add_argument('--alpha', default=0.5, type=float, help='tradeoff between loss and similarity score or confident score')
    parser.add_argument('--batch_norm', default="false", type=str, help='batch norm for the model')
    parser.add_argument('--experiment', default="tsne", type=str, help='which experiment to run')
    '''
    method: which method should be ran
        UNITY: Ours
        co_teaching
        co_teaching_plus: Disagreement
        jocor: Agreement 
        clean: AutoOD clean
        UADB: unsupervised anomaly detecter boosting
        SELFIE: selfie method 
        SEAL: seal method
        noisy_label_generator: Isolation Forest
    label_noise: how should initial noisy labels be generated
        ensemble_mv: use the majority vote of unsupervised anomaly detectors in the ensemble
        isolation_forest: use isolation forest
        knn: use knn
        lof: use lof
    strategy: how should samples be selected for update. For the co-teaching methods 
        default: use the default strategy only select small loss samples
        split: split predicted inliers and outliers and sample from both sets
    experiment: which Unity experiment should be ran
        Ablation Study 
            disagree_only: only use disagreement
            agree_only: only use agreement
            agree_and_disagree: both agree and disagree without label correction
            refurbishment_only: only use label correction
            unity: both agree and disagree with label correction
        varying_noise_rate: synthetic data
        complexity: complexity analysis, varying n. Use dataset mammography_n where n is the number of samples
        Extra 
            avg_noise: instead of giving classification methods access to GT noise rate, given them access to the average noise rates
            no_threshold: run unity without adaptive thresholding: use GT noise rate instead
            no_EMA: run unity without EMA 
            tsne: store data for tsne plots for each sample

    '''
    # setting seeds
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    print(args)

    if torch.cuda.is_available():
        print(f"Running on {torch.cuda.get_device_name(0)}")

    if args.experiment != "varying_noise_rate":
        # load in dataset
        X, y, dim, outlier_ratio = load_data(args)

        # get initial noisy labels
        y_noisy, outlier_score, autoOD_L, noise_rate_outlier, noise_rate_inlier = run_detectors(args, X, y, outlier_ratio)

        if args.experiment == "avg_noise":
            # use average outlier and inlier noise rates
            noise_rate_outlier = 0.7  # average outlier noise rate: 0.7
            noise_rate_inlier = 0.1  # average inlier noise rate: 0.1

    else:
        X, y, dim, outlier_ratio, y_noisy, noise_rate_outlier, noise_rate_inlier = load_synthetic_data(dataset=args.dataset, col_name_labels=args.col_name_labels)
        outlier_score = 0
        autoOD_L = 0

    args.batch_size = X.shape[0]

    if args.experiment == "no_EMA":
        args.beta = 0
    
    print(f"Noise Rate Outliers: {noise_rate_outlier}")
    print(f"Noise Rate Inliers: {noise_rate_inlier}")

    # create dataloader
    dataloader, X = create_dataloader(args, X, y)

    # run selected method
    if args.method == "noisy_label_generator":
        record_noisy_labels(args, y, y_noisy, outlier_score)
    
    if args.method == "UNITY":
        run_unity(args, dataloader, dim, y_noisy, outlier_ratio, outlier_score, noise_rate_outlier, noise_rate_inlier)

    if args.method == "clean":
        run_autood_clean(X, y, autoOD_L, y_noisy, args, ratio_to_remove=0.02, max_iteration=20)

    if args.method == "UADB":
        run_UADB(args, X, y, y_noisy, dataloader, dim, outlier_score)

    if args.method == "co_teaching":
        run_coteaching(args, dataloader, dim, y_noisy, outlier_ratio, noise_rate_outlier, noise_rate_inlier)

    if args.method == "co_teaching_plus":
        run_coteaching_plus(args, dataloader, dim, y_noisy, outlier_ratio, noise_rate_outlier, noise_rate_inlier)

    if args.method == "jocor":
        run_jocor(args, dataloader, dim, y_noisy, outlier_ratio, noise_rate_outlier, noise_rate_inlier)

    if args.method == "SELFIE":
        run_selfie(args, dataloader, dim, y_noisy, noise_rate_outlier, noise_rate_inlier)
    
    if args.method == "SEAL":
        run_seal(args, dataloader, dim, y_noisy)

