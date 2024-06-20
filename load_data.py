from random import seed
from re import T
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

class Custom_Dataset():

    def __init__(self, X, y, index = None, org_index=False):
        self.X, self.y = X, y
        self.org_index = org_index

        if index is None:
            self.index = np.arange(len(X))
        else:
            self.index = index

    def __getitem__(self, idx):
        if self.org_index:
            index = self.index[idx]
        else:
            index = idx
        return index, self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.index)


def load_data(args):
    '''
    This function loads in the dataset

    Parameters
    ----------
    dataset: name of dataset we want to read in
    col_name_labels: name of target feature 
    
    Returns
    -------
    X: training data
    y: GT
    dim: dimension of the dataset
    '''
    dataset = args.dataset
    col_name_labels = args.col_name_labels
    print(f"loading in {dataset} data")
    dataset = dataset.lower()
    
    if args.experiment == "complexity":
        dataset_name = dataset.split('_')[0]
        n = int(dataset.split('_')[1])
        data1 = pd.read_csv(f"datasets/large/{dataset_name}_large.csv")  # read in dataset

        # randomly sample n samples from the dataset
        data1 = data1.sample(n=n, random_state=0)
        data1 = data1.reset_index(drop=True)
    
    else:
        data1 = pd.read_csv(f"datasets/{dataset}.csv")  # read in dataset
    
    data1 = pd.DataFrame(data1)
    data1.rename(columns={col_name_labels: "label"}, inplace=True)  # rename label col from input dataset to "label"
    print(f"data shape: {data1.shape}")
    original_n = data1.shape[0]  # original dimension of the dataset

    # remove duplicate samples with different labels, keep first instance of each duplicate with same label
    features_to_exclude = ['id', 'label']
    columns_for_duplicates = data1.columns.difference(features_to_exclude)
    duplicate_indices = data1.duplicated(subset=columns_for_duplicates, keep=False)
    data_to_drop = data1[duplicate_indices]
    data1 = data1.drop(index=data_to_drop.index.difference(data_to_drop.drop_duplicates(subset=columns_for_duplicates).index))
    data1 = data1.reset_index(drop=True)
    
    print(data1.head())
    print(f"data shape: {data1.shape}")
    new_n = data1.shape[0]  # new dimension of the dataset
    print(f"Number of duplicates: {original_n - new_n}")

    # convert all labels to 0 for inlier, 1 for outlier
    if data1['label'][0] == b'no' or data1['label'][0] == b'yes':
        data1["label"] = data1["label"].map(lambda x: 1 if x == b'yes' else 0).values
    elif data1['label'][0] == b'1' or data1['label'][0] == b'0':
        data1["label"] = data1["label"].map(lambda x: 1 if x == b'1' else 0).values
    else:
        data1["label"] = data1["label"].map(lambda x: 1 if x == 1 else 0).values

    X = data1.drop(columns=['id', 'label'])
    dim = X.shape[1]  # dimension of the dataset
    y = data1["label"]

    print(f"Outlier ratio for {dataset}: {data1['label'].mean()}")
    return X.values, y.values, dim, data1['label'].mean()


def create_dataloader(args, X, y):
    '''
    This function create the dataloader used for training

    Parameters
    ----------
    args: arguments provided by user
    X: dataset
    y: GT labels
    
    Returns
    -------
    my_dataset: dataloader used for training
    ''' 
    # standardize the dataset robustly to outliers
    transformer = RobustScaler().fit(X)
    X_transformed = transformer.transform(X)
    
    # create dataloader for training
    tensor_x = torch.tensor(X_transformed, dtype=torch.float)  # transform to torch tensor
    tensor_y = torch.tensor(y, dtype=torch.long)  # GT labels
    my_dataset_temp = Custom_Dataset(tensor_x, tensor_y)
    my_dataset = DataLoader(my_dataset_temp, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=None, pin_memory=False)

    return my_dataset, X_transformed


def load_synthetic_data(dataset="spambase", col_name_labels="label"):
    '''
    This function reads in the synthetic datasets
    
    Parameters
    -----
    dataset: name of dataset
    col_name_labels: name of target feature

    Returns
    -----
    X: dataset
    y: GT labels
    dim: dimension of the dataset
    outlier_ratio: ratio of outliers in the dataset
    synthetic_labels: noisy labels
    '''
    print(f"loading in {dataset} data")
    dataset = dataset.lower()
    data1 = pd.read_csv(f"datasets/synthetic/{dataset}.csv")  # read in dataset
    data1 = pd.DataFrame(data1)

    data1.rename(columns={col_name_labels: "label"}, inplace=True)  # rename label col from input dataset to "label"
    print(f"data shape: {data1.shape}")
    original_n = data1.shape[0]  # original dimension of the dataset

    # remove duplicate samples with different labels, keep first instance of each duplicate with same label
    features_to_exclude = ['id', 'label', 'noisy_label']
    columns_for_duplicates = data1.columns.difference(features_to_exclude)
    duplicate_indices = data1.duplicated(subset=columns_for_duplicates, keep=False)
    data_to_drop = data1[duplicate_indices]
    data1 = data1.drop(index=data_to_drop.index.difference(data_to_drop.drop_duplicates(subset=columns_for_duplicates).index))
    data1 = data1.reset_index(drop=True)
    
    print(data1.head())
    print(f"data shape: {data1.shape}")
    new_n = data1.shape[0]  # new dimension of the dataset
    print(f"Number of duplicates: {original_n - new_n}")

    # convert all labels to 0 for inlier, 1 for outlier
    if data1['label'][0] == b'no' or data1['label'][0] == b'yes':
        data1["label"] = data1["label"].map(lambda x: 1 if x == b'yes' else 0).values
    elif data1['label'][0] == b'1' or data1['label'][0] == b'0':
        data1["label"] = data1["label"].map(lambda x: 1 if x == b'1' else 0).values
    else:
        data1["label"] = data1["label"].map(lambda x: 1 if x == 1 else 0).values

    X = data1.drop(columns=['id', 'label', 'noisy_label'])
    dim = X.shape[1]  # dimension of the dataset
    y = data1["label"]
    noisy_labels = data1["noisy_label"].values

    outlier_ratio = data1['label'].mean()
    print(f"Outlier ratio for {dataset}: {outlier_ratio}")
    
    # get noise rate for the outlier class for co-teaching methods
    gt_outliers = np.where(y == 1)[0]
    noise_rate_outlier = 1-accuracy_score(y[gt_outliers], noisy_labels[gt_outliers])
    gt_inliers = np.where(y == 0)[0]
    noise_rate_inlier = 1-accuracy_score(y[gt_inliers], noisy_labels[gt_inliers])
    #noise_rate_outlier = 0.6  # average outlier noise rate: 0.6
    #noise_rate_inlier = 0.1  # average inlier noise rate: 0.1
    return X.values, y.values, dim, outlier_ratio, noisy_labels, noise_rate_outlier, noise_rate_inlier
