import torch
from torch import nn
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import math
import numpy as np
import os
import csv

from methods.models.model import model

class Trainer():
    def __init__(self, args, dim):
        self.args, self.dim = args, dim
        self.init_models()

        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_raw = nn.CrossEntropyLoss(reduction="none")


    def init_models(self):
        '''
        initialize models for JoCoR
        '''
        self.model = model(feature_dim=self.dim, hidden_dim=30, num_classes=2, args=self.args).cuda()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr_co_teaching, momentum=self.args.momentum)

    
    def inference_model(self, model, dataloader, y_noisy):
        '''
        inference the model

        Parameters
        ----------
        model: model to inference
        dataloader: data loader

        Returns
        -------
        confidences: confidence scores
        pred_labels: predicted labels
        true_labels: true labels
        '''
        self.model.eval()

        logits = []
        confidences = []
        pred_labels = []
        true_labels = []
        indices = []

        for iter_num, (idx, data, label) in enumerate(dataloader):
            data, label = data.cuda(), label.cuda()

            pred_logits = self.model(data)

            pred_prob = torch.softmax(pred_logits, dim=1)
            pred_label = torch.argmax(pred_prob, dim=1)

            logits.append(pred_logits)
            confidences.append(pred_prob[:, 1])
            pred_labels.append(pred_label)
            true_labels.append(label)
            indices.append(idx)

        logits = torch.cat(logits, dim=0).detach().cpu().numpy()
        confidences = torch.cat(confidences, dim=0).detach().cpu().numpy()
        pred_labels = torch.cat(pred_labels, dim=0).detach().cpu().numpy()
        true_labels = torch.cat(true_labels, dim=0).detach().cpu().numpy()
        indices = torch.cat(indices, dim=0).detach().cpu().numpy()

        # use model confidence as outlier score
        if self.args.experiment == "varying_noise_rate":
            noisy_outlier_ratio = label.float().mean()
        else:
            noisy_outlier_ratio = torch.argmax(y_noisy, dim=1).to(torch.float64).mean().item()
        num_outliers = math.ceil(noisy_outlier_ratio * len(confidences))
        conf_sorted = np.argsort(confidences)
        pred_outlier = conf_sorted[-num_outliers:]
        pred_labels = np.zeros(len(confidences))
        pred_labels[pred_outlier] = 1

        return confidences[np.argsort(indices)], pred_labels[np.argsort(indices)], true_labels[np.argsort(indices)], logits[np.argsort(indices)]
    
    
    def seal_sgd(self, dataloader, y_noisy):
        '''
        gradient descent for SEAL

        Parameters
        ----------
        dataloader: data loader
        y_noisy : initial noisy pseudo labels
        threshold: threshold for how many clean samples to select

        Returns
        -------
        Nothing: Will update models
        '''
        # loop through batches
        for iter_num, (idx, data, label) in enumerate(dataloader):
            data, label = data.cuda(), label.cuda()

            # pass data thought model1
            model1_pred_logits = self.model(data)
            model1_pred_prob = torch.softmax(model1_pred_logits, dim=1)
            model1_outlier_scores = model1_pred_prob[:,1]
            if self.args.experiment == "varying_noise_rate":
                noisy_outlier_ratio = label.float().mean()
            else:
                noisy_outlier_ratio = torch.argmax(y_noisy, dim=1).to(torch.float64).mean().item()
            num_outliers = math.ceil(noisy_outlier_ratio * len(idx))
            conf_sorted = torch.argsort(model1_outlier_scores)
            pred_outlier = conf_sorted[-num_outliers:]
            model1_pred_label = torch.zeros(len(idx)).cuda().to(int)
            model1_pred_label[pred_outlier] = 1

            # update model            
            loss = self.loss_fn(model1_pred_logits, y_noisy[idx])  # update model 1 with model 2's selected samples
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def train_seal(self, dataloader, y_noisy):
        '''
        Main training loop for SEAL
        
        Parameters
        ----------
        dataloader: data loader
        y_noisy : initial noisy pseudo labels 

        Returns
        -------
        Nothing: Will print scores from jocor method
        '''
        max_model1_f1 = 0
        max_model1_ROCAUC = 0 

        for epoch in range(50):
            print(f"####### Epoch {epoch} #######")
            self.model.train()

            # SGD
            self.seal_sgd(dataloader, y_noisy)

            # inference model 1
            model1_confidences, model1_pred_labels, true_labels, logits1 = self.inference_model(1, dataloader, y_noisy)

            # report performance
            model1_f1 = f1_score(true_labels, model1_pred_labels)
            print("F1 score of model1 is {:2f}".format(model1_f1))
            max_model1_f1 = max(max_model1_f1, model1_f1)
            model1_auc = roc_auc_score(true_labels, model1_confidences)
            print("ROC AUC of model1 is {:2f}".format(model1_auc))
            max_model1_ROCAUC = max(max_model1_ROCAUC, model1_auc)

        print("Iteration of SEAL Done")
        return model1_pred_labels, true_labels
        

def run_seal(args, dataloader, dim, y_noisy):
    '''
    Starts running co-teaching plus
    
    Parameters
    ----------
    args: arguments from user
    dataloader: data loader
    dim: dimension of data
    y_noisy : initial noisy pseudo labels 

    Returns
    -------
    Nothing: Will save scores from co-teaching method
    '''
    print("Running SEAL")
    iterations = 3
    predictions = y_noisy
    for i in range(iterations):
        trainer = Trainer(args, dim)  # initialize models
        predictions, true_labels = trainer.train_seal(dataloader, (torch.eye(2, device='cuda:0')[predictions]))

    # log results
    print("SEAL is done training")

    # get final metrics
    f1 = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, predictions)

    print(f"SEAL F1 score: {f1}")
    
    # write results to csv
    if args.experiment == "varying_noise_rate":
        csv_file_path = f"results/final_results/{args.experiment}_analysis/{args.experiment}_{args.dataset}.csv"
        path_exists = False
        if os.path.exists(csv_file_path):
            path_exists = True
        with open(csv_file_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            if not path_exists:
                csv_writer.writerow(["Method", "Dataset", "seed", "batch_norm", "epoch", 
                                     "agree_f1", "agree_n", "agree_precision", "agree_recall",
                                     "model1_disagree_f1", "model1_disagree_n", "model1_disagree_precision", "model1_disagree_recall",
                                     "model2_disagree_f1", "model2_disagree_n", "model2_disagree_precision", "model2_disagree_recall",
                                     "model1_ctl_f1", "model1_ctl_n", "model1_ctl_precision", "model1_ctl_recall",
                                     "model2_ctl_f1", "model2_ctl_n", "model2_ctl_precision", "model2_ctl_recall",
                                     "overall_f1", "overall_precision", "overall_recall", "overall_auc",
                                     "agree_time", "disagree_time", "refurbishment_time", "training_time"])
                
            csv_writer.writerow([args.method, args.dataset, args.seed, args.batch_norm,args.epochs,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 f1, precision, recall, auc,
                                 0, 0, 0, 0])

    else:
        csv_file_path = f"results/final_results/sota_comparison/results.csv"
        path_exists = False
        if os.path.exists(csv_file_path):
            path_exists = True
    
        with open(csv_file_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            if not path_exists:
                csv_writer.writerow(["Method", "Dataset", "seed", "batch_norm", "f1", "precision", "recall", "auc"])
            csv_writer.writerow([args.method, args.dataset, args.seed, args.batch_norm, f1, precision, recall, auc])


