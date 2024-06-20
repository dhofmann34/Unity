from torch import nn
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
import math
from torch import autograd
import torch.nn.functional as F
from sklearn.preprocessing import PowerTransformer
import wandb
import csv
import os
from torch.optim.lr_scheduler import StepLR 
from torch import linalg
import random
from torch.utils.data import DataLoader
import time

from methods.models.model import model, TwinNetwork
from methods.unity.utils import *

class Trainer():
    def __init__(self, args, dim, outlier_ratio):
        self.args, self.dim, self.outlier_ratio = args, dim, outlier_ratio
        self.init_models()

        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_raw = nn.CrossEntropyLoss(reduction="none")
        self.contrastive_loss = ContrastiveLoss()


    def init_models(self):
        '''
        Initialize models
        '''
        self.model1 = model(feature_dim=self.dim, hidden_dim=self.args.hidden_dim, num_classes=2, args=self.args).cuda()
        self.model2 = model(feature_dim=self.dim, hidden_dim=self.args.hidden_dim, num_classes=2, args=self.args).cuda()
        self.contrastive_model = TwinNetwork(feature_dim=self.dim, hidden_dim=128).cuda()

        self.optimizer1 = torch.optim.SGD(self.model1.parameters(), lr=self.args.lr_co_teaching, momentum=self.args.momentum)
        self.optimizer2 = torch.optim.SGD(self.model2.parameters(), lr=self.args.lr_co_teaching, momentum=self.args.momentum)
        self.optimizer_contrastive = torch.optim.SGD(self.contrastive_model.parameters(), lr=self.args.lr_contrastive,  momentum=self.args.momentum)


    def inference_model(self, model, dataloader, y_noisy):
        '''
        Inference the model

        Parameters
        ----------
        model: model to inference
        dataloader: data loader
        y_noisy: noisy pseudo labels

        Returns
        -------
        confidences: model confidence
        pred_labels: predicted labels
        true_labels: true labels
        logits: model logits
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
        
        # gt_outlier_rate = true_labels.mean().item()
        # num_outliers = math.ceil(gt_outlier_rate * len(idx))
        
        conf_sorted = np.argsort(confidences)
        pred_outlier = conf_sorted[-num_outliers:]
        pred_labels = np.zeros(len(confidences))
        pred_labels[pred_outlier] = 1

        return confidences[np.argsort(indices)], pred_labels[np.argsort(indices)], true_labels[np.argsort(indices)], logits[np.argsort(indices)]
    

    def inference_ctl(self, dataloader, y_noisy, model1_selected_clean, model1_selected_preds_clean):
        '''
        Inference the CTL model when running refurbishment only experiment

        Parameters
        ----------
        dataloader: data loader
        y_noisy: noisy pseudo labels
        model1_selected_clean: selected samples by model 1
        model1_selected_preds_clean: model 1's selected predictions

        Returns
        -------
        pred_labels: predicted labels
        label: true labels
        distance: distance to centroid
        '''
        
        self.contrastive_model.eval()
        
        for iter_num, (idx, data, label) in enumerate(dataloader):
            sorted_idx = idx.argsort()
            data_sorted = data[sorted_idx]

            # get selected clean predicted inliers 
            pred_inlier_idx = np.nonzero(model1_selected_preds_clean[model1_selected_clean] == 0)
            pred_inlier_idx = model1_selected_clean[pred_inlier_idx]
            data_clean = data_sorted[pred_inlier_idx.cpu()]

            # get inlier centroid 
            out_inliers = self.contrastive_model.inference(data_clean.float().cuda())
            centroid = torch.mean(out_inliers, dim=0)

            # pass all data to model
            out = self.contrastive_model.inference(data.cuda())

            # calculate distance to centroid
            distance = torch.norm(out - centroid, dim=1)

            # get number of outliers
            if self.args.experiment == "varying_noise_rate":
                noisy_outlier_ratio = label.float().mean()
            else:
                noisy_outlier_ratio = torch.argmax(y_noisy, dim=1).to(torch.float64).mean().item()
            num_outliers = math.ceil(noisy_outlier_ratio * len(distance))
            distance_sorted = np.argsort(distance.cpu().detach().numpy())
            pred_outlier = distance_sorted[-num_outliers:]
            pred_labels = np.zeros(len(distance))
            pred_labels[pred_outlier] = 1

        return pred_labels[np.argsort(sorted_idx)], label[np.argsort(sorted_idx)], distance[np.argsort(sorted_idx)]
    
    
    def ema_loss(self, epoch, loss_model1, loss_model2, idx):
        '''
        Stores and calculates the EMA of the loss

        Parameters
        ----------
        epoch: current epoch
        loss_model1: model 1's loss
        loss_model2: model 2's loss
        idx: indexes of samples in the mini batch

        Returns
        -------
        loss_model1: updated model 1's loss
        loss_model2: updated model 2's loss
        '''
        if epoch <= 0:  # dont calculate EMA for first epoch
            loss_model1 = loss_model1
            loss_model2 = loss_model2
        else:
            loss_model1 = (self.args.beta * self.model1_ema.get_loss_ema(idx)) + ((1-self.args.beta) * loss_model1.cpu())
            loss_model2 = (self.args.beta * self.model2_ema.get_loss_ema(idx)) + ((1-self.args.beta) * loss_model2.cpu())

        self.model1_ema.record_loss_ema(idx, loss_model1)  # record the similarity score for each sample
        self.model2_ema.record_loss_ema(idx, loss_model2)  # record the similarity score for each sample

        return loss_model1, loss_model2


    def get_adaptive_threshold_agreement(self, trimmed_sampling_score, gt_noise_rate, sampling_score):
        '''
        Calculates the adaptive threshold for agreement module

        Parameters
        ----------
        trimmed_sampling_score: trimmed sampling scores for agreement module
        gt_noise_rate: ground truth noise rate for non adaptive thresholding experiment
        sampling_score: sampling scores for agreement module

        Returns
        -------
        threshold: adaptive threshold
        '''        
        if self.args.experiment == "no_threshold":
            num_samples_select = int(len(sampling_score) * (1 - gt_noise_rate))
            sorted_scores, _ = torch.sort(sampling_score.view(-1))
            threshold = sorted_scores[num_samples_select-1]  # select the num_samples_select with the smallest sampling scores

        else:
            if len(trimmed_sampling_score) > 1:
                threshold = max(torch.mean(trimmed_sampling_score) - (torch.std(trimmed_sampling_score)), min(trimmed_sampling_score))
            else:
                threshold = trimmed_sampling_score
        return threshold
    

    def get_adaptive_threshold_disagreement(self, trimmed_sampling_score, gt_noise_rate, sampling_score):
        '''
        Calculates the adaptive threshold for disagreement module

        Parameters
        ----------
        trimmed_sampling_score: trimmed sampling scores for disagreement module
        gt_noise_rate: ground truth noise rate for non adaptive thresholding experiment
        sampling_score: sampling scores for disagreement module
        
        Returns
        -------
        threshold: adaptive threshold        
        '''        
        if self.args.experiment == "no_threshold":
            num_samples_select = int(len(sampling_score) * (1 - gt_noise_rate))
            sorted_scores, _ = torch.sort(sampling_score.view(-1))
            threshold = sorted_scores[num_samples_select-1]  # select the num_samples_select with the smallest sampling scores
        
        else:
            if len(trimmed_sampling_score) > 1:
                threshold = max(torch.mean(trimmed_sampling_score) - (1.5 * torch.std(trimmed_sampling_score)), min(trimmed_sampling_score))
            else:
                threshold = trimmed_sampling_score
        return threshold
    
    
    def sample_selection(self, model1_pred_logits, model2_pred_logits, idx, epoch, y_noisy, model1_pred_label, model2_pred_label, noise_rate_outlier, noise_rate_inlier):
        '''
        Performs the Agree-to-Disagree clean sample selection

        Parameters
        ----------
        model1_pred_logits: model 1's predicted logits
        model2_pred_logits: model 2's predicted logits
        idx: indexes of samples in the mini batch
        epoch: current epoch
        y_noisy: noisy pseudo labels
        model1_pred_label: model 1's predicted labels
        model2_pred_label: model 2's predicted labels
        noise_rate_outlier: noise rate of outliers
        noise_rate_inlier: noise rate of inliers
        
        Returns
        -------
        model1_select_idx: selected samples by model 1
        model2_select_idx: selected samples by model 2
        model1_agree_idx: model 1's agreement predictions
        model2_agree_idx: model 2's agreement predictions
        model1_disagree_idx: model 1's disagreement predictions
        model2_disagree_idx: model 2's disagreement predictions
        agreement_time: time taken to select samples for agreement
        disagreement_time: time taken to select samples for disagreement
        model1_before_agree_idx: model 1's indexes before agreement
        model2_before_agree_idx: model 2's indexes before agreement
        model1_before_disagree_idx: model 1's indexes before disagreement
        model2_before_disagree_idx: model 2's indexes before disagreement
        '''
        # time each sample selection step
        if not epoch < self.args.epochs_warmup:
            start_time = time.time()  # agree start

        # loss
        loss_model1 = self.loss_fn_raw(model1_pred_logits, y_noisy[idx])
        loss_model2 = self.loss_fn_raw(model2_pred_logits, y_noisy[idx])

        # ema of loss
        loss_model1, loss_model2 = self.ema_loss(epoch, loss_model1, loss_model2, idx)
        loss_model1 = loss_model1.cuda()
        loss_model2 = loss_model2.cuda()
        
        # warmup by selecting all samples
        if epoch < self.args.epochs_warmup:
            end_time = time.time()  # dont time warmup
            agreement_time = 0
            disagreement_time = 0
            model1_select_idx = torch.arange(0, len(idx)).cuda()  # select all data
            model2_select_idx = torch.arange(0, len(idx)).cuda()  

            model1_agree_idx = model1_select_idx
            model2_agree_idx = model2_select_idx
            model1_disagree_idx = model1_select_idx
            model2_disagree_idx = model2_select_idx
            model1_before_agree_idx = model1_select_idx
            model2_before_agree_idx = model2_select_idx
            model1_before_disagree_idx = model1_select_idx
            model2_before_disagree_idx = model2_select_idx
        
        # Do sample selection
        else:
            alpha = self.args.alpha  # tradeoff between loss and similarity score
            if self.args.experiment != "disagree_only":

                ###### Agreement module ######
                model1_before_agree_idx = torch.arange(0, len(idx))
                model2_before_agree_idx = torch.arange(0, len(idx))
                
                # JS divergence: smaller value has more similar distributions
                m = (model1_pred_logits.softmax(1) + model2_pred_logits.softmax(1))/2
                kl_main = F.kl_div(model1_pred_logits.log_softmax(1), m, reduction='none')
                kl_weight = F.kl_div(model2_pred_logits.log_softmax(1), m, reduction='none')
                similarity_score = 0.5 * torch.sum(kl_main, 1) + 0.5 * torch.sum(kl_weight, 1)

                # Normalize similarity score
                similarity_score = (similarity_score - torch.min(similarity_score)) / (torch.max(similarity_score) - torch.min(similarity_score))  # normalize

                # split predicted inliers and outliers
                combined_inlier_idx = torch.nonzero((model1_pred_label == 0) & (model2_pred_label == 0)).flatten()
                combined_outlier_idx = torch.nonzero((model1_pred_label == 1) & (model2_pred_label == 1)).flatten()

                # loss
                combined_loss = loss_model1 + loss_model2
                combined_loss = (combined_loss - torch.min(combined_loss)) / (torch.max(combined_loss) - torch.min(combined_loss))  # normalize
                
                loss_model1 = (loss_model1 - torch.min(loss_model1)) / (torch.max(loss_model1) - torch.min(loss_model1))  # normalize
                loss_model2 = (loss_model2 - torch.min(loss_model2)) / (torch.max(loss_model2) - torch.min(loss_model2))  # normalize

                # calculate the sampling score for agreement
                sampling_score = (alpha * similarity_score) + ((1-alpha) * combined_loss)  # we want samples with smaller sampling scores
                
                # Transform the sampling score distribution to gaussian distribution
                if len(sampling_score[combined_inlier_idx]) > 1:
                    sampling_score[combined_inlier_idx] = torch.clamp(sampling_score[combined_inlier_idx], .000000001, 10)
                    power_transformer = PowerTransformer(method='box-cox')
                    sampling_score[combined_inlier_idx] = torch.from_numpy(power_transformer.fit_transform(sampling_score[combined_inlier_idx].detach().cpu().reshape(-1,1)).astype(np.float32)).flatten().cuda()
                
                # remove the outliers
                if len(sampling_score[combined_inlier_idx]) > 2:
                    # trim the outliers 
                    num_trim = math.ceil(sampling_score[combined_inlier_idx].numel() * .05)
                    _, indices_smallest = torch.topk(sampling_score[combined_inlier_idx], num_trim, largest=False)
                    _, indices_largest = torch.topk(sampling_score[combined_inlier_idx], num_trim, largest=True)
                    drop = torch.cat([indices_smallest, indices_largest])
                    mask = ~torch.isin(torch.arange(sampling_score[combined_inlier_idx].size(0)).cuda(), drop)
                    scores_trimmed_inliers = sampling_score[combined_inlier_idx][mask]

                    # sorted_scores, _ = torch.sort(sampling_score[combined_inlier_idx].view(-1))
                    # scores_trimmed_inliers = sorted_scores[num_trim:-num_trim]
                else:
                    scores_trimmed_inliers = sampling_score[combined_inlier_idx]

                if len(sampling_score[combined_outlier_idx]) > 1:
                    # Transform the sampling score distribution to gaussian distribution
                    sampling_score[combined_outlier_idx] = torch.clamp(sampling_score[combined_outlier_idx], .000000001, 10)
                    power_transformer = PowerTransformer(method='box-cox')
                    sampling_score[combined_outlier_idx] = torch.from_numpy(power_transformer.fit_transform(sampling_score[combined_outlier_idx].detach().cpu().reshape(-1,1)).astype(np.float32)).flatten().cuda()
                
                # remove the outliers
                if len(sampling_score[combined_outlier_idx]) > 2:
                    # trim the outliers 
                    num_trim = math.ceil(sampling_score[combined_outlier_idx].numel() * .05)
                    _, indices_smallest = torch.topk(sampling_score[combined_outlier_idx], num_trim, largest=False)
                    _, indices_largest = torch.topk(sampling_score[combined_outlier_idx], num_trim, largest=True)
                    drop = torch.cat([indices_smallest, indices_largest])
                    mask = ~torch.isin(torch.arange(sampling_score[combined_outlier_idx].size(0)).cuda(), drop)
                    scores_trimmed_outliers = sampling_score[combined_outlier_idx][mask]
                    
                    # sorted_scores, _ = torch.sort(sampling_score[combined_outlier_idx].view(-1))
                    # scores_trimmed_outliers = sorted_scores[num_trim:-num_trim]
                else:
                    scores_trimmed_outliers = sampling_score[combined_outlier_idx]

                # adaptive threshold for how many samples to select as clean
                sample_score_agreement_inliers_threshold = self.get_adaptive_threshold_agreement(scores_trimmed_inliers, 0.1, sampling_score[combined_inlier_idx])
                sample_score_agreement_outliers_threshold = self.get_adaptive_threshold_agreement(scores_trimmed_outliers, 0.7, sampling_score[combined_outlier_idx])
                            
                # select samples that are smaller than the threshold
                similar_smallest_losses_inliers_idx = torch.nonzero(sampling_score[combined_inlier_idx] <= sample_score_agreement_inliers_threshold).flatten()
                similar_smallest_losses_outliers_idx = torch.nonzero(sampling_score[combined_outlier_idx] <= sample_score_agreement_outliers_threshold).flatten()

                # selected indexes 
                model1_select_idx = torch.concat((combined_inlier_idx[similar_smallest_losses_inliers_idx], combined_outlier_idx[similar_smallest_losses_outliers_idx]))
                model2_select_idx = torch.concat((combined_inlier_idx[similar_smallest_losses_inliers_idx], combined_outlier_idx[similar_smallest_losses_outliers_idx]))

                model1_agree_idx = model1_select_idx
                model2_agree_idx = model2_select_idx

                if self.args.experiment == "agree_only":      
                    model1_disagree_idx = torch.empty(0, dtype=torch.long)
                    model2_disagree_idx = torch.empty(0, dtype=torch.long)
                    disagreement_time = 0
                    model1_before_disagree_idx = torch.empty(0, dtype=torch.long)
                    model2_before_disagree_idx = torch.empty(0, dtype=torch.long)
                                
            end_time = time.time()  # agree end
            agreement_time = end_time - start_time
            
            if self.args.experiment != "agree_only":
                ###### Disagreement module ######
                if self.args.experiment == "disagree_only":
                    model1_select_idx = torch.empty(0, dtype=torch.long).cuda()
                    model2_select_idx = torch.empty(0, dtype=torch.long).cuda()
                    model1_before_agree_idx = torch.empty(0, dtype=torch.long).cuda()
                    model2_before_agree_idx = torch.empty(0, dtype=torch.long).cuda()
                    agreement_time = 0
                
                model1_before_disagree_idx = torch.nonzero(~torch.isin(torch.arange(len(idx)).cuda(), model1_select_idx)).flatten()
                model2_before_disagree_idx = torch.nonzero(~torch.isin(torch.arange(len(idx)).cuda(), model2_select_idx)).flatten()
                start_time = time.time()  # disagree start
                
                # prob of sample being an outlier
                prob_model1 = model1_pred_logits.softmax(1)[:,1]
                prob_model2 = model2_pred_logits.softmax(1)[:,1]

                # calculate sampling scores. Large difference in confidence and small loss have large sampling scores
                confscore_model1 = (torch.abs(prob_model1 - .5) - torch.abs(prob_model2 - .5))
                confscore_model2 = (torch.abs(prob_model2 - .5) - torch.abs(prob_model1 - .5))

                # normalize the conf scores 
                if confscore_model1.sum().item() != self.args.batch_size and confscore_model1.sum().item() != 0:
                    confscore_model1 = (confscore_model1 - torch.min(confscore_model1)) / (torch.max(confscore_model1) - torch.min(confscore_model1))
                if confscore_model2.sum().item() != self.args.batch_size and confscore_model2.sum().item() != 0:
                    confscore_model2 = (confscore_model2 - torch.min(confscore_model2)) / (torch.max(confscore_model2) - torch.min(confscore_model2))

                # calculate sampling score for disagreement     
                large_conf_small_loss_score_model1 = (alpha * (1-confscore_model1)) + ((1 - alpha) * (loss_model1))
                large_conf_small_loss_score_model2 = (alpha * (1-confscore_model2)) + ((1 - alpha) * (loss_model2))

                largest_similarity_score_idx = torch.nonzero(~torch.isin(torch.arange(len(idx)).cuda(), model1_select_idx)).flatten()  # non-selected samples by agreement

                # split non-selected samples into pred inliers and outliers
                largest_similarity_score_idx_model1_inlier = torch.tensor(np.intersect1d(torch.nonzero(model1_pred_label== 0).flatten().cpu(), largest_similarity_score_idx.cpu())).cuda()
                largest_similarity_score_idx_model1_outlier = torch.tensor(np.intersect1d(torch.nonzero(model1_pred_label== 1).flatten().cpu(), largest_similarity_score_idx.cpu())).cuda()
                largest_similarity_score_idx_model2_inlier = torch.tensor(np.intersect1d(torch.nonzero(model2_pred_label== 0).flatten().cpu(), largest_similarity_score_idx.cpu())).cuda() 
                largest_similarity_score_idx_model2_outlier = torch.tensor(np.intersect1d(torch.nonzero(model2_pred_label== 1).flatten().cpu(), largest_similarity_score_idx.cpu())).cuda()

                if len(largest_similarity_score_idx_model1_inlier) > 1:
                    # Transform the sampling score distribution to gaussian distribution
                    large_conf_small_loss_score_model1[largest_similarity_score_idx_model1_inlier] = torch.clamp(large_conf_small_loss_score_model1[largest_similarity_score_idx_model1_inlier], .000000001, 10)
                    power_transformer = PowerTransformer(method='box-cox')
                    large_conf_small_loss_score_model1_inlier = torch.from_numpy(power_transformer.fit_transform(large_conf_small_loss_score_model1[largest_similarity_score_idx_model1_inlier].detach().cpu().reshape(-1,1)).astype(np.float32)).flatten().cuda()
                else:
                    large_conf_small_loss_score_model1_inlier = large_conf_small_loss_score_model1[largest_similarity_score_idx_model1_inlier]

                # remove the outliers
                if len(large_conf_small_loss_score_model1_inlier) > 2:
                    # trim the outliers 
                    num_trim = math.ceil(large_conf_small_loss_score_model1_inlier.numel() * .05)
                    _, indices_smallest = torch.topk(large_conf_small_loss_score_model1_inlier, num_trim, largest=False)
                    _, indices_largest = torch.topk(large_conf_small_loss_score_model1_inlier, num_trim, largest=True)
                    drop = torch.cat([indices_smallest, indices_largest])
                    mask = ~torch.isin(torch.arange(large_conf_small_loss_score_model1_inlier.size(0)).cuda(), drop)
                    scores_trimmed_inliers_model1 = large_conf_small_loss_score_model1_inlier[mask]

                else:
                    scores_trimmed_inliers_model1 = large_conf_small_loss_score_model1_inlier

                if len(largest_similarity_score_idx_model1_outlier) > 1:
                    large_conf_small_loss_score_model1[largest_similarity_score_idx_model1_outlier] = torch.clamp(large_conf_small_loss_score_model1[largest_similarity_score_idx_model1_outlier], .000000001, 10)
                    power_transformer = PowerTransformer(method='box-cox')
                    large_conf_small_loss_score_model1_outlier = torch.from_numpy(power_transformer.fit_transform(large_conf_small_loss_score_model1[largest_similarity_score_idx_model1_outlier].detach().cpu().reshape(-1,1)).astype(np.float32)).flatten().cuda()
                else:
                    large_conf_small_loss_score_model1_outlier = large_conf_small_loss_score_model1[largest_similarity_score_idx_model1_outlier]

                # remove the outliers
                if len(large_conf_small_loss_score_model1_outlier) > 2:
                    # trim the outliers 
                    num_trim = math.ceil(large_conf_small_loss_score_model1_outlier.numel() * .05)
                    _, indices_smallest = torch.topk(large_conf_small_loss_score_model1_outlier, num_trim, largest=False)
                    _, indices_largest = torch.topk(large_conf_small_loss_score_model1_outlier, num_trim, largest=True)
                    drop = torch.cat([indices_smallest, indices_largest])
                    mask = ~torch.isin(torch.arange(large_conf_small_loss_score_model1_outlier.size(0)).cuda(), drop)
                    scores_trimmed_outliers_model1 = large_conf_small_loss_score_model1_outlier[mask]
                    
                else:
                    scores_trimmed_outliers_model1 = large_conf_small_loss_score_model1_outlier

                if len(largest_similarity_score_idx_model2_inlier) > 1:
                    large_conf_small_loss_score_model2[largest_similarity_score_idx_model2_inlier] = torch.clamp(large_conf_small_loss_score_model2[largest_similarity_score_idx_model2_inlier], .000000001, 10)
                    power_transformer = PowerTransformer(method='box-cox')
                    large_conf_small_loss_score_model2_inlier = torch.from_numpy(power_transformer.fit_transform(large_conf_small_loss_score_model2[largest_similarity_score_idx_model2_inlier].detach().cpu().reshape(-1,1)).astype(np.float32)).flatten().cuda()
                else:
                    large_conf_small_loss_score_model2_inlier = large_conf_small_loss_score_model2[largest_similarity_score_idx_model2_inlier]

                # remove the outliers
                if len(large_conf_small_loss_score_model2_inlier) > 2:
                    # trim the outliers 
                    num_trim = math.ceil(large_conf_small_loss_score_model2_inlier.numel() * .05)
                    _, indices_smallest = torch.topk(large_conf_small_loss_score_model2_inlier, num_trim, largest=False)
                    _, indices_largest = torch.topk(large_conf_small_loss_score_model2_inlier, num_trim, largest=True)
                    drop = torch.cat([indices_smallest, indices_largest])
                    mask = ~torch.isin(torch.arange(large_conf_small_loss_score_model2_inlier.size(0)).cuda(), drop)
                    scores_trimmed_inliers_model2 = large_conf_small_loss_score_model2_inlier[mask]
                    
                else:
                    scores_trimmed_inliers_model2 = large_conf_small_loss_score_model2_inlier
                
                if len(largest_similarity_score_idx_model2_outlier) > 1:
                    large_conf_small_loss_score_model2[largest_similarity_score_idx_model2_outlier] = torch.clamp(large_conf_small_loss_score_model2[largest_similarity_score_idx_model2_outlier], .000000001, 10)
                    power_transformer = PowerTransformer(method='box-cox')
                    large_conf_small_loss_score_model2_outlier = torch.from_numpy(power_transformer.fit_transform(large_conf_small_loss_score_model2[largest_similarity_score_idx_model2_outlier].detach().cpu().reshape(-1,1)).astype(np.float32)).flatten().cuda()
                else:
                    large_conf_small_loss_score_model2_outlier = large_conf_small_loss_score_model2[largest_similarity_score_idx_model2_outlier]

                # remove the outliers
                if len(large_conf_small_loss_score_model2_outlier) > 2:
                    # trim the outliers 
                    num_trim = math.ceil(large_conf_small_loss_score_model2_outlier.numel() * .05)
                    _, indices_smallest = torch.topk(large_conf_small_loss_score_model2_outlier, num_trim, largest=False)
                    _, indices_largest = torch.topk(large_conf_small_loss_score_model2_outlier, num_trim, largest=True)
                    drop = torch.cat([indices_smallest, indices_largest])
                    mask = ~torch.isin(torch.arange(large_conf_small_loss_score_model2_outlier.size(0)).cuda(), drop)
                    scores_trimmed_outliers_model2 = large_conf_small_loss_score_model2_outlier[mask]
                   
                else:
                    scores_trimmed_outliers_model2 = large_conf_small_loss_score_model2_outlier
            
                # adaptive threshold
                large_conf_sample_score_model1_inlier_threshold = self.get_adaptive_threshold_disagreement(scores_trimmed_inliers_model1, 0.1, large_conf_small_loss_score_model1_inlier)
                large_conf_sample_score_model1_outlier_threshold = self.get_adaptive_threshold_disagreement(scores_trimmed_outliers_model1, 0.7, large_conf_small_loss_score_model1_outlier)
                large_conf_sample_score_model2_inlier_threshold = self.get_adaptive_threshold_disagreement(scores_trimmed_inliers_model2, 0.1, large_conf_small_loss_score_model2_inlier)
                large_conf_sample_score_model2_outlier_threshold = self.get_adaptive_threshold_disagreement(scores_trimmed_outliers_model2, 0.7, large_conf_small_loss_score_model2_outlier)

                # select samples that are smaller than the threshold
                large_conf_sample_score_model1_inlier_idx = torch.nonzero(large_conf_small_loss_score_model1_inlier <= large_conf_sample_score_model1_inlier_threshold).flatten()
                large_conf_sample_score_model1_outlier_idx = torch.nonzero(large_conf_small_loss_score_model1_outlier <= large_conf_sample_score_model1_outlier_threshold).flatten()
                large_conf_sample_score_model2_inlier_idx = torch.nonzero(large_conf_small_loss_score_model2_inlier <= large_conf_sample_score_model2_inlier_threshold).flatten()
                large_conf_sample_score_model2_outlier_idx = torch.nonzero(large_conf_small_loss_score_model2_outlier <= large_conf_sample_score_model2_outlier_threshold).flatten()

                model1_select_idx = torch.cat((model1_select_idx, largest_similarity_score_idx_model1_inlier[large_conf_sample_score_model1_inlier_idx], largest_similarity_score_idx_model1_outlier[large_conf_sample_score_model1_outlier_idx]))
                model2_select_idx = torch.cat((model2_select_idx, largest_similarity_score_idx_model2_inlier[large_conf_sample_score_model2_inlier_idx], largest_similarity_score_idx_model2_outlier[large_conf_sample_score_model2_outlier_idx]))

                model1_disagree_idx = torch.cat([largest_similarity_score_idx_model1_inlier[large_conf_sample_score_model1_inlier_idx], largest_similarity_score_idx_model1_outlier[large_conf_sample_score_model1_outlier_idx]])
                model2_disagree_idx = torch.cat([largest_similarity_score_idx_model2_inlier[large_conf_sample_score_model2_inlier_idx], largest_similarity_score_idx_model2_outlier[large_conf_sample_score_model2_outlier_idx]])

                if self.args.experiment == "disagree_only":
                    model1_agree_idx = torch.empty(0, dtype=torch.long)
                    model2_agree_idx = torch.empty(0, dtype=torch.long)
            
                end_time = time.time()  # disagree end 
                disagreement_time = end_time - start_time
        return model1_select_idx, model2_select_idx, model1_agree_idx, model2_agree_idx, model1_disagree_idx, model2_disagree_idx, agreement_time, disagreement_time, model1_before_agree_idx, model2_before_agree_idx, model1_before_disagree_idx, model2_before_disagree_idx
        

    def unity_sgd(self, dataloader, y_noisy, epoch, outlier_score, noise_rate_outlier, noise_rate_inlier):
        '''
        Gradient descent for UNITY

        Parameters
        ----------
        dataloader: data loader
        y_noisy: noisy pseudo labels
        epoch: current epoch
        outlier_score: isolation forest outlier scores
        noise_rate_outlier: noise rate of outliers
        noise_rate_inlier: noise rate of inliers
        
        Returns
        -------
        y_noisy: updated noisy pseudo labels
        model1_selected_clean: selected clean samples by model 1
        model2_selected_clean: selected clean samples by model 2
        model1_selected_preds_clean: selected clean samples' predicted labels by model 1
        model2_selected_preds_clean: selected clean samples' predicted labels by model 2
        model1_agreement_pred: model 1's agreement predictions
        model1_agreement_gt: model 1's agreement ground truth
        model2_agreement_pred: model 2's agreement predictions
        model2_agreement_gt: model 2's agreement ground truth
        model1_disagreement_pred: model 1's disagreement predictions
        model1_disagreement_gt: model 1's disagreement ground truth
        model2_disagreement_pred: model 2's disagreement predictions
        model2_disagreement_gt: model 2's disagreement ground truth
        contrastive_pred_1: contrastive model's predictions for model 1
        contrastive_pred_2: contrastive model's predictions for model 2
        contrastive_gt_1: contrastive model's ground truth for model 1
        contrastive_gt_2: contrastive model's ground truth for model 2
        model1_before_agree_pred: model 1's predictions before agreement
        model1_before_agree_gt: model 1's ground truth before agreement
        model2_before_agree_pred: model 2's predictions before agreement
        model2_before_agree_gt: model 2's ground truth before agreement
        model1_before_disagree_pred: model 1's predictions before disagreement
        model1_before_disagree_gt: model 1's ground truth before disagreement
        model2_before_disagree_pred: model 2's predictions before disagreement
        model2_before_disagree_gt: model 2's ground truth before disagreement
        before_refurb_1: model 1's predictions before refurbishment
        before_refurb_2: model 2's predictions before refurbishment
        agreement_time: time taken for agreement module
        disagreement_time: time taken for disagreement module
        refurbishment_time: time taken for refurbishment module
        training_time: time taken for training
        '''
        # track each module's performance 
        model1_agreement_pred = []
        model1_agreement_gt = []
        model2_agreement_pred = []
        model2_agreement_gt = []
        model1_disagreement_pred = []
        model1_disagreement_gt = []
        model2_disagreement_pred = []
        model2_disagreement_gt = []
        contrastive_pred_1 = []
        contrastive_pred_2 = []
        contrastive_gt_1 = []
        contrastive_gt_2 = []
        model1_before_agree_pred = []
        model1_before_agree_gt = []
        model2_before_agree_pred = []
        model2_before_agree_gt = []
        model1_before_disagree_pred = []
        model1_before_disagree_gt = []
        model2_before_disagree_pred = []
        model2_before_disagree_gt = []
        before_refurb_1 = []
        before_refurb_2 = []

        model1_selected_clean = []
        model2_selected_clean = []
        model1_selected_preds_clean = np.full(len(y_noisy), -1)
        model2_selected_preds_clean = np.full(len(y_noisy), -1)

        # loop through batches
        for iter_num, (idx, data, label) in enumerate(dataloader):
            data, label = data.cuda(), label.cuda()
            
            if self.args.experiment == "refurbishment_only":
                # get most confident samples from isolation forest 
                outlier_score = outlier_score[idx]
                self.args.epochs_warmup = 0
                outlier_score_idx = np.argsort(outlier_score)
                num_conf = .2
                num_outliers_conf = math.ceil(len(idx) * num_conf * label.float().mean().item())
                num_inliers_conf = math.ceil(len(idx) * num_conf * (1-label.float().mean().item()))
                outlier_idx = outlier_score_idx[-num_outliers_conf:]
                inlier_idx = outlier_score_idx[:num_inliers_conf]
                
                model1_pred_label = y_noisy.argmax(dim=1)[idx]
                model2_pred_label = y_noisy.argmax(dim=1)[idx]
                model1_select_idx = torch.tensor(np.concatenate((outlier_idx, inlier_idx))).cuda()
                model2_select_idx = torch.tensor(np.concatenate((outlier_idx, inlier_idx))).cuda()
                model1_selected_preds_clean = model1_pred_label
                model1_agree_idx = [[]]
                model1_pred_logits = [[]]
                model2_pred_logits = [[]]
                model1_selected_clean = model1_select_idx
                model2_selected_clean = model2_select_idx

                agreement_time = 0
                disagreement_time = 0
                training_time = 0
            
            else:
                model1_pred_logits = self.model1(data)
                model1_pred_prob = torch.softmax(model1_pred_logits, dim=1)
                model1_outlier_scores = model1_pred_prob[:,1]
                if self.args.experiment == "varying_noise_rate":
                    noisy_outlier_ratio = label.float().mean()
                else:
                    noisy_outlier_ratio = torch.argmax(y_noisy, dim=1)[idx].to(torch.float64).mean().item()
                num_outliers = math.ceil(noisy_outlier_ratio * len(idx))                
                conf_sorted = torch.argsort(model1_outlier_scores)
                pred_outlier = conf_sorted[-num_outliers:]
                model1_pred_label = torch.zeros(len(idx)).cuda().to(int)
                model1_pred_label[pred_outlier] = 1

                model2_pred_logits = self.model2(data)
                model2_pred_prob = torch.softmax(model2_pred_logits, dim=1)
                model2_outlier_scores = model2_pred_prob[:,1]
                num_outliers = math.ceil(noisy_outlier_ratio * len(idx))
                conf_sorted = torch.argsort(model2_outlier_scores)
                pred_outlier = conf_sorted[-num_outliers:]
                model2_pred_label = torch.zeros(len(idx)).cuda().to(int)
                model2_pred_label[pred_outlier] = 1   

                # store the predicted labels for historical predictions
                self.model1_preds_hist.record_preds(idx, model1_outlier_scores)
                self.model2_preds_hist.record_preds(idx, model1_outlier_scores)
                
                # Sample selection
                model1_select_idx, model2_select_idx, model1_agree_idx, model2_agree_idx, model1_disagree_idx, model2_disagree_idx, agreement_time, disagreement_time, model1_before_agree_idx, model2_before_agree_idx, model1_before_disagree_idx, model2_before_disagree_idx = self.sample_selection(model1_pred_logits, model2_pred_logits, idx, epoch, y_noisy, model1_pred_label, model2_pred_label, noise_rate_outlier, noise_rate_inlier)

                model1_selected_preds_clean[idx[model1_select_idx.cpu()]] = model1_pred_label[model1_select_idx].cpu()
                model2_selected_preds_clean[idx[model2_select_idx.cpu()]] = model2_pred_label[model2_select_idx].cpu()
                model1_selected_clean.append(idx[model1_select_idx.cpu()])
                model2_selected_clean.append(idx[model2_select_idx.cpu()])            
                
                # track performance of each module
                model1_agreement_pred.append(model1_pred_label[model1_agree_idx])
                model2_agreement_pred.append(model2_pred_label[model2_agree_idx])
                model1_disagreement_pred.append(model1_pred_label[model1_disagree_idx])
                model2_disagreement_pred.append(model2_pred_label[model2_disagree_idx])
                model1_before_agree_pred.append(model1_pred_label[model1_before_agree_idx])
                model2_before_agree_pred.append(model2_pred_label[model2_before_agree_idx])
                model1_before_disagree_pred.append(model1_pred_label[model1_before_disagree_idx])
                model2_before_disagree_pred.append(model2_pred_label[model2_before_disagree_idx])

                model1_agreement_gt.append(label[model1_agree_idx])
                model2_agreement_gt.append(label[model2_agree_idx])
                model1_disagreement_gt.append(label[model1_disagree_idx])
                model2_disagreement_gt.append(label[model2_disagree_idx])
                model1_before_agree_gt.append(label[model1_before_agree_idx])
                model2_before_agree_gt.append(label[model2_before_agree_idx])
                model1_before_disagree_gt.append(label[model1_before_disagree_idx])
                model2_before_disagree_gt.append(label[model2_before_disagree_idx])

            # Label Correction               
            if (self.args.experiment == "unity" or self.args.experiment == "varying_noise_rate" or self.args.experiment == "refurbishment_only" or self.args.experiment == "no_threshold" or self.args.experiment == "complexity" or self.args.experiment == "no_EMA" or self.args.experiment == "tsne"):
                if epoch >= self.args.epochs_warmup:
                    model1_pred_label_before = model1_pred_label.clone()
                    model2_pred_label_before = model2_pred_label.clone()
                    model1_select_idx, model2_select_idx, model1_pred_label, model2_pred_label, model1_corr_labels_idx, model2_corr_labels_idx, model1_gt_contrastive, model2_gt_contrastive, refurbishment_time = self.update_labels_ctl(model1_pred_label, model2_pred_label, y_noisy, idx, model1_select_idx, model2_select_idx, data, epoch, model1_agree_idx, label, model1_pred_logits, model2_pred_logits)
                    
                    # track accuracy of contrastive correction
                    contrastive_pred_1.append(model1_pred_label[model1_corr_labels_idx])
                    contrastive_pred_2.append(model2_pred_label[model2_corr_labels_idx])
                    contrastive_gt_1.append(model1_gt_contrastive)
                    contrastive_gt_2.append(model2_gt_contrastive)
                    before_refurb_1.append(model1_pred_label_before[model1_corr_labels_idx])
                    before_refurb_2.append(model2_pred_label_before[model2_corr_labels_idx])

                else:
                    contrastive_pred_1.append(model1_pred_label[model1_select_idx])
                    contrastive_pred_2.append(model2_pred_label[model2_select_idx])
                    contrastive_gt_1.append(label[model1_select_idx])
                    contrastive_gt_2.append(label[model2_select_idx])
                    refurbishment_time = 0
                    before_refurb_1.append(model1_pred_label[model1_select_idx])
                    before_refurb_2.append(model2_pred_label[model2_select_idx])
            else:
                contrastive_pred_1.append(model1_pred_label[model1_select_idx])
                contrastive_pred_2.append(model2_pred_label[model2_select_idx])
                contrastive_gt_1.append(label[model1_select_idx])
                contrastive_gt_2.append(label[model2_select_idx])
                refurbishment_time = 0
                before_refurb_1.append(model1_pred_label[model1_select_idx])
                before_refurb_2.append(model2_pred_label[model2_select_idx])
            
            if epoch < self.args.epochs_warmup:
                model1_cf_data, model1_cf_label = data[model1_select_idx], torch.argmax(y_noisy[idx], axis=1)[model1_select_idx]
                model2_cf_data, model2_cf_label = data[model2_select_idx], torch.argmax(y_noisy[idx], axis=1)[model2_select_idx]

            else:  # use model predictions
                # shuffle the selected indices
                model1_select_idx = model1_select_idx[torch.randperm(len(model1_select_idx))]
                model2_select_idx = model2_select_idx[torch.randperm(len(model2_select_idx))]
                
                if self.args.experiment != "unity" and self.args.experiment != "varying_noise_rate" and self.args.experiment != "no_threshold" and self.args.experiment != "complexity" and self.args.experiment != "tsne" and self.args.experiment != "no_EMA":
                    # do only selection: No refurbishment
                    model1_cf_data, model1_cf_label = data[model1_select_idx], torch.argmax(y_noisy[idx], axis=1)[model1_select_idx]  
                    model2_cf_data, model2_cf_label = data[model2_select_idx], torch.argmax(y_noisy[idx], axis=1)[model2_select_idx]

                else:
                    model1_cf_data, model1_cf_label = data[model1_select_idx], model1_pred_label[model1_select_idx]  
                    model2_cf_data, model2_cf_label = data[model2_select_idx], model2_pred_label[model2_select_idx]
            
            if self.args.experiment != "refurbishment_only":
                start_time = time.time()  # training start 
                # Update model 1 with model 2 confident samples
                co_model1_loss = self.loss_fn(model1_pred_logits[model2_select_idx], model2_cf_label)
                self.optimizer1.zero_grad()
                co_model1_loss.backward()
                self.optimizer1.step()

                # Update model 2 with model 1 confident samples
                co_model2_loss = self.loss_fn(model2_pred_logits[model1_select_idx], model1_cf_label)
                self.optimizer2.zero_grad()
                co_model2_loss.backward()
                self.optimizer2.step()

                end_time = time.time()  # training end
                training_time = end_time - start_time

        
        if self.args.experiment == "tsne" and epoch >= self.args.epochs_warmup:  # write each samples selection to file
            # model1_agree_idx, model1_disagree_idx, model1_corr_labels_idx
            n = len(idx)
            selection = np.full(n, -1)
            selection[model1_agree_idx.cpu()] = 1
            selection[model1_disagree_idx.cpu()] = 2
            selection[model1_corr_labels_idx.cpu()] = 3

            tsne_df = pd.DataFrame({'ds' : [self.args.dataset]*n,
                                    'batch_norm' : [self.args.batch_norm]*n,
                                    'seed' : [self.args.seed]*n, 
                                    'epoch' : [epoch]*n, 
                                    'id': idx, 
                                    'model1_pred' : model1_pred_label[idx].cpu().numpy(), 
                                    'model2_pred': model2_pred_label[idx].cpu().numpy(), 
                                    'noisy_label': y_noisy.argmax(dim=1)[idx][idx].cpu().numpy(), 
                                    'gt': label[idx].cpu().numpy(), 
                                    'selection': selection[idx],
                                    'label_before_refub': model1_pred_label_before[idx].cpu().numpy(),
                                    'label_after_refub': model1_pred_label[idx].cpu().numpy(),
                                   })

            # write to csv
            csv_file_path = f"results/final_results/system_evaluation/evaluation_{self.args.experiment}_{self.args.dataset}.csv"
            if os.path.exists(csv_file_path):
                tsne_df.to_csv(csv_file_path, mode='a', index=False, header=False)
            else:
                tsne_df.to_csv(csv_file_path, mode='w', index=False, header=True)

        return y_noisy, model1_agreement_pred, model2_agreement_pred, model1_agreement_gt, model2_agreement_gt, model1_disagreement_pred, model2_disagreement_pred, model1_disagreement_gt, model2_disagreement_gt, model1_selected_clean, model2_selected_clean, model1_selected_preds_clean, contrastive_pred_1, contrastive_pred_2, contrastive_gt_1, contrastive_gt_2, agreement_time, disagreement_time, refurbishment_time, training_time, model1_before_agree_pred, model1_before_agree_gt, model2_before_agree_pred, model2_before_agree_gt, model1_before_disagree_pred, model1_before_disagree_gt, model2_before_disagree_pred, model2_before_disagree_gt, before_refurb_1, before_refurb_2


    def train_unity(self, dataloader, y_noisy, outlier_score, noise_rate_outlier, noise_rate_inlier):
        '''
        Main training loop for Unity
        
        Parameters
        ----------
        dataloader: data loader
        y_noisy : initial noisy pseudo
        outlier_score: outlier scores from isolation forest
        noise_rate_outlier: noise rate of outliers
        noise_rate_inlier: noise rate of inliers

        Returns
        -------
        Nothing: Will print and log performance metrics of Unity
        '''        
        # Store history of losses and confidences
        self.model1_ema = EMARecorder(y_noisy)
        self.model2_ema = EMARecorder(y_noisy)
        self.initial_label_constant = y_noisy.clone()
        self.model1_preds_hist = PredsRecorder(self.args.epochs, y_noisy)
        self.model2_preds_hist = PredsRecorder(self.args.epochs, y_noisy)
       
        for epoch in range(self.args.epochs):
            print(" ")
            print(f"####### Epoch {epoch} #######")
            self.model1.train()
            self.model2.train()
            self.model1_preds_hist.update_epoch(epoch)
            self.model2_preds_hist.update_epoch(epoch)

            # SGD
            y_noisy_updated, model1_agreement_pred, model2_agreement_pred, model1_agreement_gt, model2_agreement_gt, model1_disagreement_pred, model2_disagreement_pred, model1_disagreement_gt, model2_disagreement_gt, model1_selected_clean, model2_selected_clean, model1_selected_preds_clean, contrastive_pred_1, contrastive_pred_2, contrastive_gt_1, contrastive_gt_2, agreement_time, disagreement_time, refurbishment_time, training_time, model1_before_agree_pred, model1_before_agree_gt, model2_before_agree_pred, model2_before_agree_gt, model1_before_disagree_pred, model1_before_disagree_gt, model2_before_disagree_pred, model2_before_disagree_gt, before_refurb_1, before_refurb_2 = self.unity_sgd(dataloader, y_noisy, epoch, outlier_score, noise_rate_outlier, noise_rate_inlier)
            y_noisy = y_noisy_updated

            if self.args.experiment != "refurbishment_only":
                model1_selected_clean = torch.cat(model1_selected_clean, dim=0)
                model2_selected_clean = torch.cat(model2_selected_clean, dim=0)
                      
                # inference model 1
                model1_confidences, model1_pred_labels, true_labels, logits1 = self.inference_model(1, dataloader, y_noisy)

                # inference model 2
                # model2_confidences, model2_pred_labels, true_labels, logits2 = self.inference_model(2, dataloader, y_noisy)          
            
                # report performance
                model1_f1 = f1_score(true_labels, model1_pred_labels)
                model1_precision = precision_score(true_labels, model1_pred_labels) 
                model1_recall = recall_score(true_labels, model1_pred_labels)
                print(" ")
                print(f"F1 score of model1 is {model1_f1}")
                model1_auc = roc_auc_score(true_labels, model1_confidences)
                print(f"ROC AUC of model1 is {model1_auc}")
                # model2_f1 = f1_score(true_labels, model2_pred_labels)        
                # print(f"F1 score of model2 is {model2_f1}")
                # model2_auc = roc_auc_score(true_labels, model2_confidences)
                # print(f"ROC AUC of model2 is {model2_auc}")
                print(" ")
                print(" ")
                if self.args.experiment != "disagree_only":
                    model1_agreementf1 = f1_score(torch.cat(model1_agreement_gt, dim=0).cpu(), torch.cat(model1_agreement_pred, dim=0).cpu())
                    model2_agreementf1 = f1_score(torch.cat(model2_agreement_gt, dim=0).cpu(), torch.cat(model2_agreement_pred, dim=0).cpu())
                    print(f"F1 score of agreement model1: {model1_agreementf1}")
                    print(f"F1 score of agreement model2: {model2_agreementf1}")
                if self.args.experiment != "agree_only":
                    model1_disagreementf1 = f1_score(torch.cat(model1_disagreement_gt, dim=0).cpu(), torch.cat(model1_disagreement_pred, dim=0).cpu())
                    model2_disagreementf1 = f1_score(torch.cat(model2_disagreement_gt, dim=0).cpu(), torch.cat(model2_disagreement_pred, dim=0).cpu())
                    print(f"F1 score of disagreement model1: {model1_disagreementf1}")
                    print(f"F1 score of disagreement model2: {model2_disagreementf1}")

            else:
                # inference ctl model
                pred_labels_ctl, label_ctl, ctl_scores = self.inference_ctl(dataloader, y_noisy, model1_selected_clean, model1_selected_preds_clean)

                f1_score_ctl = f1_score(label_ctl.cpu(), pred_labels_ctl)
                precision_score_ctl = precision_score(label_ctl.cpu(), pred_labels_ctl)
                recall_score_ctl = recall_score(label_ctl.cpu(), pred_labels_ctl)
                auc_score_ctl = roc_auc_score(label_ctl.cpu(), ctl_scores.detach().cpu())
                print(f"F1 score of ctl model: {f1_score_ctl}")

            # write results to CSV
            if self.args.experiment == "unity" and epoch >= self.args.epochs_warmup:
                # write system evaluation to csv
                csv_file_path = f"results/final_results/system_evaluation/evaluation_{self.args.experiment}_{self.args.dataset}.csv"
                path_exists = False
                if os.path.exists(csv_file_path):
                    path_exists = True
                with open(csv_file_path, mode='a', newline='') as file:
                    csv_writer = csv.writer(file)
                    if not path_exists:
                        csv_writer.writerow(["Method", "Dataset", "seed", "batch_norm", "epoch", 
                                            "agree_f1", "agree_n", "agree_precision", "agree_recall", "before_agree_f1", "before_agree_n", "before_agree_precision", "before_agree_recall",
                                            "model1_disagree_f1", "model1_disagree_n", "model1_disagree_precision", "model1_disagree_recall", "model1_before_disagree_f1", "model1_before_disagree_n", "model1_before_disagree_precision", "model1_before_disagree_recall",
                                            "model2_disagree_f1", "model2_disagree_n", "model2_disagree_precision", "model2_disagree_recall", "model2_before_disagree_f1", "model2_before_disagree_n", "model2_before_disagree_precision", "model2_before_disagree_recall",
                                            "model1_ctl_f1", "model1_ctl_n", "model1_ctl_precision", "model1_ctl_recall", "model1_before_ctl_f1", "model1_before_ctl_n", "model1_before_ctl_precision", "model1_before_ctl_recall",
                                            "model2_ctl_f1", "model2_ctl_n", "model2_ctl_precision", "model2_ctl_recall", "model2_before_ctl_f1", "model2_before_ctl_n", "model2_before_ctl_precision", "model2_before_ctl_recall",
                                            "overall_f1", "overall_precision", "overall_recall", "overall_auc",
                                            "agree_time", "disagree_time", "refurbishment_time", "training_time"])
                        
                    csv_writer.writerow([self.args.method, self.args.dataset, self.args.seed, self.args.batch_norm, epoch,
                                        model1_agreementf1, len(model1_agreement_pred[0]), precision_score(torch.cat(model1_agreement_gt, dim=0).cpu(), torch.cat(model1_agreement_pred, dim=0).cpu()), recall_score(torch.cat(model1_agreement_gt, dim=0).cpu(), torch.cat(model1_agreement_pred, dim=0).cpu()), f1_score(model1_before_agree_gt[0].cpu(), model1_before_agree_pred[0].cpu()), len(model1_before_agree_pred[0]), precision_score(torch.cat(model1_before_agree_gt, dim=0).cpu(), torch.cat(model1_before_agree_pred, dim=0).cpu()), recall_score(torch.cat(model1_before_agree_gt, dim=0).cpu(), torch.cat(model1_before_agree_pred, dim=0).cpu()),
                                        model1_disagreementf1, len(model1_disagreement_pred[0]), precision_score(torch.cat(model1_disagreement_gt, dim=0).cpu(), torch.cat(model1_disagreement_pred, dim=0).cpu()), recall_score(torch.cat(model1_disagreement_gt, dim=0).cpu(), torch.cat(model1_disagreement_pred, dim=0).cpu()), f1_score(model1_before_disagree_gt[0].cpu(), model1_before_disagree_pred[0].cpu()), len(model1_before_disagree_pred[0]), precision_score(torch.cat(model1_before_disagree_gt, dim=0).cpu(), torch.cat(model1_before_disagree_pred, dim=0).cpu()), recall_score(torch.cat(model1_before_disagree_gt, dim=0).cpu(), torch.cat(model1_before_disagree_pred, dim=0).cpu()),
                                        model2_disagreementf1, len(model2_disagreement_pred[0]), precision_score(torch.cat(model2_disagreement_gt, dim=0).cpu(), torch.cat(model2_disagreement_pred, dim=0).cpu()), recall_score(torch.cat(model2_disagreement_gt, dim=0).cpu(), torch.cat(model2_disagreement_pred, dim=0).cpu()), f1_score(model2_before_disagree_gt[0].cpu(), model2_before_disagree_pred[0].cpu()), len(model2_before_disagree_pred[0]), precision_score(torch.cat(model2_before_disagree_gt, dim=0).cpu(), torch.cat(model2_before_disagree_pred, dim=0).cpu()), recall_score(torch.cat(model2_before_disagree_gt, dim=0).cpu(), torch.cat(model2_before_disagree_pred, dim=0).cpu()),
                                        f1_score(contrastive_gt_1[0].cpu(), contrastive_pred_1[0].cpu()), len(contrastive_pred_1[0]), precision_score(contrastive_gt_1[0].cpu(), contrastive_pred_1[0].cpu()), recall_score(contrastive_gt_1[0].cpu(), contrastive_pred_1[0].cpu()), f1_score(contrastive_gt_1[0].cpu(), before_refurb_1[0].cpu()), len(contrastive_gt_1[0]), precision_score(contrastive_gt_1[0].cpu(), before_refurb_1[0].cpu()), recall_score(contrastive_gt_1[0].cpu(), before_refurb_1[0].cpu()),
                                        f1_score(contrastive_gt_2[0].cpu(), contrastive_pred_2[0].cpu()), len(contrastive_pred_2[0]), precision_score(contrastive_gt_2[0].cpu(), contrastive_pred_2[0].cpu()), recall_score(contrastive_gt_2[0].cpu(), contrastive_pred_2[0].cpu()), f1_score(contrastive_gt_2[0].cpu(), before_refurb_2[0].cpu()), len(contrastive_gt_2[0]), precision_score(contrastive_gt_2[0].cpu(), before_refurb_2[0].cpu()), recall_score(contrastive_gt_2[0].cpu(), before_refurb_2[0].cpu()),
                                        model1_f1, model1_precision, model1_recall, model1_auc,
                                        agreement_time, disagreement_time, refurbishment_time, training_time])
                    
            elif (self.args.experiment == "complexity" and epoch >= self.args.epochs_warmup) or (self.args.experiment == "varying_noise_rate" and epoch >= self.args.epochs_warmup) or (self.args.experiment == "no_threshold" and epoch >= self.args.epochs_warmup) or (self.args.experiment == "no_EMA" and epoch >= self.args.epochs_warmup):
                # write system evaluation to csv
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
                        
                    csv_writer.writerow([self.args.method, self.args.dataset, self.args.seed, self.args.batch_norm, epoch,
                                        model1_agreementf1, len(model1_agreement_pred[0]), precision_score(torch.cat(model1_agreement_gt, dim=0).cpu(), torch.cat(model1_agreement_pred, dim=0).cpu()), recall_score(torch.cat(model1_agreement_gt, dim=0).cpu(), torch.cat(model1_agreement_pred, dim=0).cpu()),
                                        model1_disagreementf1, len(model1_disagreement_pred[0]), precision_score(torch.cat(model1_disagreement_gt, dim=0).cpu(), torch.cat(model1_disagreement_pred, dim=0).cpu()), recall_score(torch.cat(model1_disagreement_gt, dim=0).cpu(), torch.cat(model1_disagreement_pred, dim=0).cpu()),
                                        model2_disagreementf1, len(model2_disagreement_pred[0]), precision_score(torch.cat(model2_disagreement_gt, dim=0).cpu(), torch.cat(model2_disagreement_pred, dim=0).cpu()), recall_score(torch.cat(model2_disagreement_gt, dim=0).cpu(), torch.cat(model2_disagreement_pred, dim=0).cpu()),
                                        f1_score(contrastive_gt_1[0].cpu(), contrastive_pred_1[0].cpu()), len(contrastive_pred_1[0]), precision_score(contrastive_gt_1[0].cpu(), contrastive_pred_1[0].cpu()), recall_score(contrastive_gt_1[0].cpu(), contrastive_pred_1[0].cpu()),
                                        f1_score(contrastive_gt_2[0].cpu(), contrastive_pred_2[0].cpu()), len(contrastive_pred_2[0]), precision_score(contrastive_gt_2[0].cpu(), contrastive_pred_2[0].cpu()), recall_score(contrastive_gt_2[0].cpu(), contrastive_pred_2[0].cpu()),
                                        model1_f1, model1_precision, model1_recall, model1_auc,
                                        agreement_time, disagreement_time, refurbishment_time, training_time])
                                                        
        print("Done training Unity!")
        
        # write results to csv
        # normal run
        if self.args.experiment == "unity":
            csv_file_path = f"results/final_results/sota_comparison/results.csv"
            path_exists = False
            if os.path.exists(csv_file_path):
                path_exists = True
            with open(csv_file_path, mode='a', newline='') as file:
                csv_writer = csv.writer(file)
                if not path_exists:
                    csv_writer.writerow(["Method", "Dataset", "seed", "batch_norm", "f1", "precision", "recall", "auc"])
                csv_writer.writerow([self.args.method, self.args.dataset, self.args.seed, self.args.batch_norm, model1_f1, model1_precision, model1_recall, model1_auc])
        
        # ablation study
        elif self.args.experiment == "refurbishment_only" or self.args.experiment == "disagree_only" or self.args.experiment == "agree_only" or self.args.experiment == "agree_and_disagree":
            csv_file_path = f"results/final_results/ablation_study/results_ablation.csv"
            path_exists = False
            if os.path.exists(csv_file_path):
                path_exists = True
            with open(csv_file_path, mode='a', newline='') as file:
                csv_writer = csv.writer(file)
                if not path_exists:
                    csv_writer.writerow(["Method", "Exp", "Dataset", "seed", "batch_norm", "f1", "precision", "recall", "auc"])
                if self.args.experiment == "refurbishment_only":
                    model1_f1 = f1_score_ctl
                    model1_precision = precision_score_ctl
                    model1_recall = recall_score_ctl
                    model1_auc = auc_score_ctl
                    
                csv_writer.writerow([self.args.method, self.args.experiment ,self.args.dataset, self.args.seed, self.args.batch_norm, model1_f1, model1_precision, model1_recall, model1_auc])          


    def update_labels_ctl(self, model1_preds, model2_preds, y_noisy, idx, model1_selected_clean_idx, model2_selected_clean_idx, data, epoch, model1_agree_idx, gt_labels, model1_pred_logits, model2_pred_logits):
        '''
        ContrastCorr Label refurbishment: Use the samples with clean labels from agree and disagree to train a contrastive learning model
        Then use the contrastive model to correct the labels of the remaining non-selected samples
        
        Parameters
        ----------
        model1_preds: model 1 predictions
        model2_preds: model 2 predictions
        y_noisy: noisy pseudo labels
        idx: indices of the samples
        model1_selected_clean_idx: indices of the selected clean samples for model 1
        model2_selected_clean_idx: indices of the selected clean samples for model 2
        data: data
        epoch: current epoch
        model1_agree_idx: indices of the samples that agree
        gt_labels: ground truth labels
        model1_pred_logits: model 1 logits
        model2_pred_logits: model 2 logits

        Returns
        -------
        model1_selected_clean_idx: indices of the selected clean samples for model 1
        model2_selected_clean_idx: indices of the selected clean samples for model 2
        model1_preds: model 1 predictions
        model2_preds: model 2 predictions
        model1_corr_labels_idx: indices of the samples that were corrected by the contrastive model
        model2_corr_labels_idx: indices of the samples that were corrected by the contrastive model
        model1_gt_contrastive: ground truth labels of the samples that were corrected by the contrastive model
        model2_gt_contrastive: ground truth labels of the samples that were corrected by the contrastive model
        refurbishment_time: time taken to train the contrastive model
        '''
        start_time = time.time()  # refurbishment start
        # training set
        pred_inliers_idx_1 = np.nonzero((model1_preds[model1_selected_clean_idx] == 0)).flatten()  # clean predicted inliers for training 
        pred_outliers_idx_1 = np.nonzero((model1_preds[model1_selected_clean_idx] == 1)).flatten()  # all predicted outliers and selected as clean samples for training 
        pred_inliers_idx_1 = model1_selected_clean_idx[pred_inliers_idx_1]
        pred_outliers_idx_1 = model1_selected_clean_idx[pred_outliers_idx_1]

        pred_inliers_idx_2 = np.nonzero((model2_preds[model2_selected_clean_idx] == 0)).flatten()  # clean predicted inliers for training
        pred_outliers_idx_2 = np.nonzero((model2_preds[model2_selected_clean_idx] == 1)).flatten()  # all predicted outliers and selected as clean samples for training
        pred_inliers_idx_2 = model2_selected_clean_idx[pred_inliers_idx_2]
        pred_outliers_idx_2 = model2_selected_clean_idx[pred_outliers_idx_2]
        
        pred_inliers_idx = torch.concatenate((pred_inliers_idx_1, pred_inliers_idx_2))
        pred_outliers_idx = torch.concatenate((pred_outliers_idx_1, pred_outliers_idx_2))
        pred_inliers_idx = torch.unique(pred_inliers_idx)
        pred_outliers_idx = torch.unique(pred_outliers_idx)

        # inference set
        non_selected_1_idx = np.nonzero(~np.isin(np.arange(len(data)), model1_selected_clean_idx.cpu()))[0]
        non_selected_2_idx = np.nonzero(~np.isin(np.arange(len(data)), model2_selected_clean_idx.cpu()))[0]

        model1_before_refurb_idx = non_selected_1_idx
        model2_before_refurb_idx = non_selected_2_idx
        
        # Prepare positive pairs (inlier, inlier) and negative pairs (inlier, outlier)
        data = data.cpu()
        pred_inliers_idx = pred_inliers_idx.cpu()
        pred_outliers_idx = pred_outliers_idx.cpu()
        n_pairs = len(pred_inliers_idx)
        
        pos_samples_idx = np.random.choice(pred_inliers_idx, n_pairs, replace=False)

        check = np.where(pred_inliers_idx == pos_samples_idx)[0]
        while check.size != 0:
            pos_samples_idx = np.random.choice(pred_inliers_idx, n_pairs, replace=False)
            check = np.where(pred_inliers_idx == pos_samples_idx)[0]

        neg_samples_idx = pred_outliers_idx.repeat(n_pairs // pred_outliers_idx.shape[0] + 1)[:n_pairs]
        neg_samples_idx = neg_samples_idx[torch.randperm(neg_samples_idx.size(0))]

        positive_pairs = torch.stack((data[pred_inliers_idx], data[pos_samples_idx]), dim=1)
        negative_pairs = torch.stack((data[pred_inliers_idx], data[neg_samples_idx]), dim=1)

        pairs = torch.concat([positive_pairs, negative_pairs])
        labels = torch.cat((torch.zeros(len(pred_inliers_idx)), torch.ones(len(pred_inliers_idx))))
        
        # Create DataLoader
        train_dataset = CustomDatasetCont(pairs, labels)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # train contrastive model
        # if epoch < 25 or epoch % 5 == 0:
        self.contrastive_model.train()
        for epoch in range(1):
            for batch_idx, (pair, label) in enumerate(train_loader):
                self.optimizer_contrastive.zero_grad()
                output1, output2 = self.contrastive_model(pair[:, 0, :].float().cuda(), pair[:, 1, :].float().cuda())
                loss = self.contrastive_loss(output1, output2, label.cuda())
                loss.backward()
                self.optimizer_contrastive.step()
        
        self.contrastive_model.eval()
        # get selected clean inlier embeddings to calculate the centroid
        output_inliers, output2 = self.contrastive_model(data[pred_inliers_idx].float().cuda(), data[pred_inliers_idx].float().cuda())
        training_inlier_embedding = output_inliers.detach().cpu().numpy()  # only get the embedding of the inliers
        centroid = np.mean(training_inlier_embedding, axis=0)  # calculate the centroid of the selected inliers

        # pass non-selected samples through the contrastive model
        output1_1, output2_1 = self.contrastive_model(data[non_selected_1_idx].float().cuda(), data[non_selected_1_idx].float().cuda())  # get new embedding
        non_selected_dists_1 = np.linalg.norm(output1_1.detach().cpu().numpy() - centroid, axis=1)  # calculate the distance of each non-selected sample from the centroid

        output1_2, output2_2 = self.contrastive_model(data[non_selected_2_idx].float().cuda(), data[non_selected_2_idx].float().cuda())  # get new embedding
        non_selected_dists_2 = np.linalg.norm(output1_2.detach().cpu().numpy() - centroid, axis=1)

        # get number of outliers
        noisy_outlier_ratio_1 = torch.argmax(y_noisy[idx][non_selected_1_idx], dim=1).to(torch.float64).mean().item()
        num_outliers_1 = math.ceil(len(output1_1) * noisy_outlier_ratio_1)
        num_inliers_1 = len(output1_1) - num_outliers_1
        noisy_outlier_ratio_2 = torch.argmax(y_noisy[idx][non_selected_2_idx], dim=1).to(torch.float64).mean().item()
        num_outliers_2 = math.ceil(len(output1_2) * noisy_outlier_ratio_2)
        num_inliers_2 = len(output1_2) - num_outliers_2

        # number of samples to ignore
        num_ignore_outliers_1 = math.ceil(num_outliers_1 * .4)
        num_ignore_inliers_1 = math.ceil(num_inliers_1 * .4)
        num_ignore_outliers_2 = math.ceil(num_outliers_2 * .4)
        num_ignore_inliers_2 = math.ceil(num_inliers_2 * .4)
        
        # sort the distances
        sorted_outputs_idx_1 = np.argsort(non_selected_dists_1)  # increasing order
        outlier_idx_1 = sorted_outputs_idx_1[-(num_outliers_1-num_ignore_outliers_1):]
        inlier_idx_1 = sorted_outputs_idx_1[:(num_inliers_1-num_ignore_inliers_1)]
        outlier_idx_1 = non_selected_1_idx[outlier_idx_1]
        inlier_idx_1 = non_selected_1_idx[inlier_idx_1]

        sorted_outputs_idx_2 = np.argsort(non_selected_dists_2)  # increasing order
        outlier_idx_2 = sorted_outputs_idx_2[-(num_outliers_2-num_ignore_outliers_2):]
        inlier_idx_2 = sorted_outputs_idx_2[:(num_inliers_2-num_ignore_inliers_2)]
        outlier_idx_2 = non_selected_2_idx[outlier_idx_2]
        inlier_idx_2 = non_selected_2_idx[inlier_idx_2]

        # contrastive learning labels for all non-selected samples
        contrastive_pred_1 = model1_preds.clone()
        contrastive_pred_1[outlier_idx_1] = 1
        contrastive_pred_1[inlier_idx_1] = 0

        contrastive_pred_2 = model2_preds.clone()
        contrastive_pred_2[outlier_idx_2] = 1
        contrastive_pred_2[inlier_idx_2] = 0

        # correct the labels
        model1_corr_labels_idx = torch.cat((torch.tensor(inlier_idx_1), torch.tensor(outlier_idx_1))).cuda()
        model2_corr_labels_idx = torch.cat((torch.tensor(inlier_idx_2), torch.tensor(outlier_idx_2))).cuda()

        model1_preds[inlier_idx_1] = contrastive_pred_1[inlier_idx_1]
        model1_preds[outlier_idx_1] = contrastive_pred_1[outlier_idx_1]
        model2_preds[inlier_idx_2] = contrastive_pred_2[inlier_idx_2]
        model2_preds[outlier_idx_2] = contrastive_pred_2[outlier_idx_2]

        end_time = time.time()  # refurbishment end
        time_refurbishment = end_time - start_time

        print(f"F1 Score of contrastive learning model 1: {f1_score(gt_labels[model1_corr_labels_idx].cpu(), model1_preds[model1_corr_labels_idx].cpu())}")
        print(f"F1 Score of contrastive learning model 2: {f1_score(gt_labels[model2_corr_labels_idx].cpu(), model1_preds[model2_corr_labels_idx].cpu())}")

        model1_select_idx = torch.cat((model1_selected_clean_idx, model1_corr_labels_idx))
        model2_select_idx = torch.cat((model2_selected_clean_idx, model2_corr_labels_idx))
        model1_select_idx = torch.unique(model1_select_idx)
        model2_select_idx = torch.unique(model2_select_idx)

        return model1_select_idx, model2_select_idx, model1_preds, model2_preds, model1_corr_labels_idx, model2_corr_labels_idx, gt_labels[model1_corr_labels_idx], gt_labels[model2_corr_labels_idx], time_refurbishment
    

def run_unity(args, dataloader, dim, y_noisy, outlier_ratio, outlier_score, noise_rate_outlier, noise_rate_inlier):
    '''
    Starts running Unity
    
    Parameters
    ----------
    args: arguments
    dataloader: data loader
    dim: dimension of the data
    y_noisy: initial noisy pseudo labels
    outlier_ratio: ratio of outliers
    outlier_score: outlier scores from isolation forest
    noise_rate_outlier: noise rate of outliers
    noise_rate_inlier: noise rate of inliers
    
    Returns
    -------
    Nothing: Will save scores from running unity
    '''
    print(f"Running Unity with experiemnt: {args.experiment}")
    trainer = Trainer(args, dim, outlier_ratio)  # initialize models
    trainer.train_unity(dataloader, (torch.eye(2, device='cuda:0')[y_noisy]), outlier_score, noise_rate_outlier, noise_rate_inlier)  # train models