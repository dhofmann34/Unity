from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
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
        initialize models for JoCoR
        '''
        self.model1 = model(feature_dim=self.dim, hidden_dim=30, num_classes=2, args=self.args).cuda()
        self.model2 = model(feature_dim=self.dim, hidden_dim=30, num_classes=2, args=self.args).cuda()

        self.optimizer1 = torch.optim.SGD(self.model1.parameters(), lr=self.args.lr_co_teaching, momentum=self.args.momentum)  # lr was 1e-3, weight_decay=1e-4
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


    def jocor_sgd(self, dataloader, y_noisy, threshold_outlier, threshold_inlier, outlier_ratio):
        '''
        gradient descent for JoCoR

        Parameters
        ----------
        dataloader: data loader
        y_noisy : initial noisy pseudo labels
        threshold: threshold for how many clean samples to select

        Returns
        -------
        Nothing: Will update models
        '''
        alpha = .5  # how much weight to give to loss vs similarity score
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
            combined_loss = loss_model1 + loss_model2

            # select small loss samples that are in agreement as clean
            # use KL divergence to measure similarity between two models: select the smallest samples
            kl_1 = F.kl_div(model1_pred_logits.log_softmax(1), model2_pred_logits.softmax(1), reduction='none')
            kl_2 = F.kl_div(model2_pred_logits.log_softmax(1), model1_pred_logits.softmax(1), reduction='none')
            similarity_score = torch.sum(kl_1, 1) + torch.sum(kl_2, 1)

            sampling_score = (alpha * similarity_score) + ((1-alpha) * combined_loss)

            if self.args.strategy == "default":
                num_sample = int(len(idx)*outlier_ratio)  # number of samples to select as clean
                select_idx = torch.argsort(sampling_score, descending=False)[:num_sample]
            
            elif self.args.strategy == "split":                
                # split predicted inliers and outliers
                combined_inlier_idx = torch.nonzero((model1_pred_label == 0) & (model2_pred_label == 0)).flatten()
                combined_outlier_idx = torch.nonzero((model1_pred_label == 1) & (model2_pred_label == 1)).flatten()

                num_sample_inliers = int(len(combined_inlier_idx) * threshold_inlier)
                num_sample_outliers = int(len(combined_outlier_idx) * threshold_outlier)

                combined_sampling_score_inlier = sampling_score[combined_inlier_idx]
                combined_sampling_score_outlier = sampling_score[combined_outlier_idx]

                # select small loss samples as clean
                select_idx_inlier = torch.argsort(combined_sampling_score_inlier, descending=False)[:num_sample_inliers]
                select_idx_outlier = torch.argsort(combined_sampling_score_outlier, descending=False)[:num_sample_outliers]
                
                select_idx = torch.cat((combined_inlier_idx[select_idx_inlier], combined_outlier_idx[select_idx_outlier]))

            # model updates            
            co_model1_loss = self.loss_fn(model1_pred_logits[select_idx], y_noisy[idx][select_idx])  # update model 1 with model 2's selected samples
            self.optimizer1.zero_grad()
            co_model1_loss.backward()
            self.optimizer1.step()

            co_model2_loss = self.loss_fn(model2_pred_logits[select_idx], y_noisy[idx][select_idx])  # update model 2 with model 1's selected samples
            self.optimizer2.zero_grad()
            co_model2_loss.backward()
            self.optimizer2.step()
            

    def train_jocor(self, dataloader, y_noisy, outlier_ratio, noise_rate_outlier, noise_rate_inlier):
        '''
        Main training loop for jocor
        
        Parameters
        ----------
        dataloader: data loader
        y_noisy : initial noisy pseudo labels 

        Returns
        -------
        Nothing: Will print scores from jocor method
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
            self.jocor_sgd(dataloader, y_noisy, threshold_outlier, threshold_inlier, outlier_ratio)

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

        print("Done training JoCoR: Below are the max scores")
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
        

def run_jocor(args, dataloader, dim, y_noisy, outlier_ratio, noise_rate_outlier, noise_rate_inlier):
    '''
    Starts running JoCoR
    
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
    print("Running JoCoR")
    trainer = Trainer(args, dim)  # initialize models
    trainer.train_jocor(dataloader, (torch.eye(2, device='cuda:0')[y_noisy]), outlier_ratio, noise_rate_outlier, noise_rate_inlier)  # train models