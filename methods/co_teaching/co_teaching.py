from torch import nn
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
import math
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
        initialize models for co-teaching
        '''
        self.model1 = model(feature_dim=self.dim, hidden_dim=30, num_classes=2, args=self.args).cuda()
        self.model2 = model(feature_dim=self.dim, hidden_dim=30, num_classes=2, args=self.args).cuda()

        self.optimizer1 = torch.optim.SGD(self.model1.parameters(), lr=self.args.lr_co_teaching, momentum=self.args.momentum)
        self.optimizer2 = torch.optim.SGD(self.model2.parameters(), lr=self.args.lr_co_teaching, momentum=self.args.momentum)

        #self.scheduler1 = StepLR(self.optimizer1, step_size=5, gamma=0.3)
        #self.scheduler2 = StepLR(self.optimizer2, step_size=5, gamma=0.3)


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
        self.model1.eval()
        self.model2.eval()

        logits = []
        confidences = []
        pred_labels = []
        true_labels = []
        indices = []

        for iter_num, (idx, data, label) in enumerate(dataloader):
            data, label = data.cuda(), label.cuda()

            if model == 1:
                pred_logits = self.model1(data)
            else:
                pred_logits = self.model2(data)

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


    def coteaching_sgd(self, dataloader, y_noisy, threshold_outlier, threshold_inlier, outlier_ratio):
        '''
        gradient descent for co-teaching

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
            model1_pred_logits = self.model1(data)
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

            # pass data thought model2
            model2_pred_logits = self.model2(data)
            model2_pred_prob = torch.softmax(model2_pred_logits, dim=1)
            model2_outlier_scores = model2_pred_prob[:,1]
            num_outliers = math.ceil(noisy_outlier_ratio * len(idx))
            conf_sorted = torch.argsort(model2_outlier_scores)
            pred_outlier = conf_sorted[-num_outliers:]
            model2_pred_label = torch.zeros(len(idx)).cuda().to(int)
            model2_pred_label[pred_outlier] = 1
            
            # losses
            loss_model1 = self.loss_fn_raw(model1_pred_logits.detach(), y_noisy[idx].detach())
            loss_model2 = self.loss_fn_raw(model2_pred_logits.detach(), y_noisy[idx].detach())

            if self.args.strategy == "default":
                # select small loss samples as clean
                num_sample = int(len(idx)*threshold_outlier)  # number of samples to select as clean
                model1_select_idx = torch.argsort(loss_model1, descending=False)[:num_sample]
                model2_select_idx = torch.argsort(loss_model2, descending=False)[:num_sample]
            
            elif self.args.strategy == "split":
                # split into pred inliers and outliers
                model1_inlier_idx = torch.nonzero(model1_pred_label== 0).flatten().cpu().cuda()
                model1_outlier_idx = torch.nonzero(model1_pred_label== 1).flatten().cpu().cuda()
                model2_inlier_idx = torch.nonzero(model2_pred_label== 0).flatten().cpu().cuda()
                model2_outlier_idx = torch.nonzero(model2_pred_label== 1).flatten().cpu().cuda()

                num_sample_inliers_1 = int(len(model1_inlier_idx) * threshold_inlier)
                num_sample_inliers_2 = int(len(model2_inlier_idx) * threshold_inlier)
                num_sample_outliers_1 = int(len(model1_outlier_idx) * threshold_outlier)
                num_sample_outliers_2 = int(len(model2_outlier_idx) * threshold_outlier)

                loss_model1_inlier = loss_model1[model1_inlier_idx]
                loss_model1_outlier = loss_model1[model1_outlier_idx]
                loss_model2_inlier = loss_model2[model2_inlier_idx]
                loss_model2_outlier = loss_model2[model2_outlier_idx]

                # select small loss samples as clean
                selected_model1_inlier = torch.argsort(loss_model1_inlier, descending=False)[:num_sample_inliers_1]
                selected_model1_outlier = torch.argsort(loss_model1_outlier, descending=False)[:num_sample_outliers_1]
                selected_model2_inlier = torch.argsort(loss_model2_inlier, descending=False)[:num_sample_inliers_2]
                selected_model2_outlier = torch.argsort(loss_model2_outlier, descending=False)[:num_sample_outliers_2]
                
                model1_select_idx = torch.cat((model1_inlier_idx[selected_model1_inlier], model1_outlier_idx[selected_model1_outlier]))
                model2_select_idx = torch.cat((model2_inlier_idx[selected_model2_inlier], model2_outlier_idx[selected_model2_outlier]))

    
            # model updates            
            co_model1_loss = self.loss_fn(model1_pred_logits[model2_select_idx], y_noisy[idx][model2_select_idx])  # update model 1 with model 2's selected samples
            self.optimizer1.zero_grad()
            co_model1_loss.backward()
            self.optimizer1.step()

            co_model2_loss = self.loss_fn(model2_pred_logits[model1_select_idx], y_noisy[idx][model1_select_idx])  # update model 2 with model 1's selected samples
            self.optimizer2.zero_grad()
            co_model2_loss.backward()
            self.optimizer2.step()
            

    def train_coteaching(self, dataloader, y_noisy, outlier_ratio, noise_rate_outlier, noise_rate_inlier):
        '''
        Main training loop for co-teaching
        
        Parameters
        ----------
        dataloader: data loader
        y_noisy : initial noisy pseudo labels 

        Returns
        -------
        Nothing: Will print scores from co-teaching method
        '''
        max_model1_f1 = 0
        max_model2_f1 = 0
        max_model1_ROCAUC = 0 
        max_model2_ROCAUC = 0

        for epoch in range(self.args.epochs):
            print(f"####### Epoch {epoch} #######")
            self.model1.train()
            self.model2.train()

            # co-teaching threshold for how many clean samples to select
            threshold_outlier = 1 - noise_rate_outlier * min((epoch/10), 1)
            threshold_inlier = 1 - noise_rate_inlier * min((epoch/10), 1)

            # SGD
            self.coteaching_sgd(dataloader, y_noisy, threshold_outlier, threshold_inlier, outlier_ratio)

            # inference model 1
            model1_confidences, model1_pred_labels, true_labels, logits1 = self.inference_model(1, dataloader, y_noisy)

            # inference model 2
            model2_confidences, model2_pred_labels, true_labels, logits2 = self.inference_model(2, dataloader, y_noisy)

            # report performance
            model1_f1 = f1_score(true_labels, model1_pred_labels)
            print("F1 score of model1 is {:2f}".format(model1_f1))
            max_model1_f1 = max(max_model1_f1, model1_f1)
            model1_auc = roc_auc_score(true_labels, model1_confidences)
            print("ROC AUC of model1 is {:2f}".format(model1_auc))
            max_model1_ROCAUC = max(max_model1_ROCAUC, model1_auc)

            model2_f1 = f1_score(true_labels, model2_pred_labels)
            print("F1 score of model2 is {:2f}".format(model2_f1))
            max_model2_f1 = max(max_model2_f1, model2_f1)
            model2_auc = roc_auc_score(true_labels, model2_confidences)
            print("ROC AUC of model2 is {:2f}".format(model2_auc))
            max_model2_ROCAUC = max(max_model2_ROCAUC, model2_auc)

        print("Done training co-teaching: Below are the max scores")
        print("Max F1 score of model1 is {:2f}".format(max_model1_f1))
        print("Max ROC AUC of model1 is {:2f}".format(max_model1_ROCAUC))
        print("Max F1 score of model2 is {:2f}".format(max_model2_f1))
        print("Max ROC AUC of model2 is {:2f}".format(max_model2_ROCAUC))

        # get final metrics
        f1 = f1_score(true_labels, model1_pred_labels)
        precision = precision_score(true_labels, model1_pred_labels)
        recall = recall_score(true_labels, model1_pred_labels)
        auc = roc_auc_score(true_labels, model1_confidences)
        
        # write results to csv
        if self.args.experiment == "varying_noise_rate":
            csv_file_path = f"results/final_results/{self.args.experiment}_analysis/{self.args.experiment}_{self.args.dataset}.csv"
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
                    
                csv_writer.writerow([self.args.method, self.args.dataset, self.args.seed, self.args.batch_norm, self.args.epochs,
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
                csv_writer.writerow([self.args.method, self.args.dataset, self.args.seed, self.args.batch_norm, f1, precision, recall, auc])
        

def run_coteaching(args, dataloader, dim, y_noisy, outlier_ratio, noise_rate_outlier, noise_rate_inlier):
    '''
    Starts running co-teaching
    
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
    print("Running co-teaching")
    trainer = Trainer(args, dim)  # initialize models
    trainer.train_coteaching(dataloader, (torch.eye(2, device='cuda:0')[y_noisy]), outlier_ratio, noise_rate_outlier, noise_rate_inlier)  # train models