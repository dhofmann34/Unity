from torch import nn
import torch
import numpy as np
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
from sklearn.cluster import KMeans
import warnings

from methods.models.model import model, SiameseNetwork, CTLNetwork
from methods.unity.utils import *

class Trainer():
    def __init__(self, args, dim, outlier_ratio):
        self.args, self.dim, self.outlier_ratio = args, dim, outlier_ratio
        self.init_models()

        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_raw = nn.CrossEntropyLoss(reduction="none")
        self.contrastive_loss = ContrastiveLoss()
        self.sup_contrastive_loss = SupervisedContrastiveLoss()


    def init_models(self):
        '''
        initialize models
        '''
        self.model1 = model(feature_dim=self.dim, hidden_dim=self.args.hidden_dim, num_classes=2, args=self.args).cuda()
        self.model2 = model(feature_dim=self.dim, hidden_dim=self.args.hidden_dim, num_classes=2, args=self.args).cuda()
        self.contrastive_model = SiameseNetwork(feature_dim=self.dim, hidden_dim=128).cuda()

        self.optimizer1 = torch.optim.SGD(self.model1.parameters(), lr=self.args.lr_co_teaching, momentum=self.args.momentum)
        self.optimizer2 = torch.optim.SGD(self.model2.parameters(), lr=self.args.lr_co_teaching, momentum=self.args.momentum)
        self.optimizer_contrastive = torch.optim.SGD(self.contrastive_model.parameters(), lr=self.args.lr_contrastive,  momentum=self.args.momentum)
        #self.optimizer_contrastive = torch.optim.Adam(self.contrastive_model.parameters(), lr=self.args.lr_contrastive)

        #self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=self.args.lr)
        #self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=self.args.lr)

        #self.scheduler1 = StepLR(self.optimizer1, step_size=5, gamma=0.3)
        #self.scheduler2 = StepLR(self.optimizer2, step_size=5, gamma=0.3)


    def inference_model(self, model, dataloader, y_noisy):
        '''
        inference the model

        Parameters
        ----------
        model: model to infer
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
        noisy_outlier_ratio = torch.argmax(y_noisy, dim=1).to(torch.float64).mean().item()
        num_outliers = math.ceil(noisy_outlier_ratio * len(confidences))
        
        # gt_outlier_rate = true_labels.mean().item()
        # num_outliers = math.ceil(gt_outlier_rate * len(idx))
        
        conf_sorted = np.argsort(confidences)
        pred_outlier = conf_sorted[-num_outliers:]
        pred_labels = np.zeros(len(confidences))
        pred_labels[pred_outlier] = 1

        #pred_labels = np.argmax(logits, axis=1)

        return confidences[np.argsort(indices)], pred_labels[np.argsort(indices)], true_labels[np.argsort(indices)], logits[np.argsort(indices)]
    

    def inference_ctl(self, dataloader, y_noisy, model1_selected_clean, model1_selected_preds_clean):
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
            noisy_outlier_ratio = torch.argmax(y_noisy, dim=1).to(torch.float64).mean().item()
            num_outliers = math.ceil(noisy_outlier_ratio * len(distance))
            distance_sorted = np.argsort(distance.cpu().detach().numpy())
            pred_outlier = distance_sorted[-num_outliers:]
            pred_labels = np.zeros(len(distance))
            pred_labels[pred_outlier] = 1

        return pred_labels[np.argsort(sorted_idx)], label[np.argsort(sorted_idx)]

    
    
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
    
    def ema_sim_score(self, epoch, sim_score, idx):
        '''
        Stores and calculates the EMA of the similarity score

        Parameters
        ----------
        epoch: current epoch
        sim_score: similarity score
        idx: indexes of samples in the mini batch

        Returns
        -------
        sim_score: updated similarity score
        '''
        if epoch <= 0:  # dont calculate EMA for first epoch
            sim_score = sim_score
        else:
            sim_score = (self.args.beta * self.model1_ema.get_sim_ema(idx)) + ((1-self.args.beta) * sim_score)  # model 1 and 2 are combined
        
        self.model1_ema.record_sim_ema(idx, sim_score)  # record the similarity score for each sample

        return sim_score
    
    def ema_prob(self, epoch, prob_model1, prob_model2, idx):
        '''
        Stores and calculates the EMA of the probability of a sample being an outlier

        Parameters
        ----------
        epoch: current epoch
        prob_model1: model 1's probability of a sample being an outlier
        prob_model2: model 2's probability of a sample being an outlier
        idx: indexes of samples in the mini batch

        Returns
        -------
        prob_model1: updated model 1's probability of a sample being an outlier
        prob_model2: updated model 2's probability of a sample being an outlier
        '''
        if epoch <= 0:  # dont calculate EMA for first epoch
            prob_model1 = prob_model1
            prob_model2 = prob_model2
        else: 
            prob_model1 = (self.args.beta * self.model1_ema.get_prob_ema(idx)) + ((1-self.args.beta) * prob_model1)
            prob_model2 = (self.args.beta * self.model2_ema.get_prob_ema(idx)) + ((1-self.args.beta) * prob_model2)
        
        self.model1_ema.record_prob_ema(idx, prob_model1)  # record the similarity score for each sample
        self.model2_ema.record_prob_ema(idx, prob_model2)  # record the similarity score for each sample

        return prob_model1, prob_model2
    

    def get_adaptive_threshold_agreement(self, trimmed_sampling_score, gt_noise_rate, sampling_score):
        '''
        Calculates the adaptive threshold for agreement module

        Parameters
        ----------
        sampling_score: sampling scores for agreement module
        gt_noise_rate: ground truth noise rate for non adaptive thresholding experiment  

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
                #threshold = min(max(trimmed_sampling_score), torch.mean(trimmed_sampling_score) + (torch.std(trimmed_sampling_score)))
                threshold = max(torch.mean(trimmed_sampling_score) - (1.5 * torch.std(trimmed_sampling_score)), min(trimmed_sampling_score))
            else:
                threshold = trimmed_sampling_score
        return threshold
    
    
    def sample_selection(self, model1_pred_logits, model2_pred_logits, idx, epoch, y_noisy, model1_pred_label, model2_pred_label, noise_rate_outlier, noise_rate_inlier):
        '''
        Given a mini batch of data each model selects samples that have clean labels

        Parameters
        ----------
        model1_pred_logits: model 1's predictions
        model2_pred_logits: model 2's predictions
        idx: indexes of samples in the mini batch
        epoch: current epoch
        y_noisy : initial noisy pseudo labels
        model1_pred_label: model 1's predicted labels
        model2_pred_label: model 2's predicted labels

        Returns
        -------
        model1_select_idx: indexes of samples selected by model 1
        model2_select_idx: indexes of samples selected by model 2
        '''
        # loss
        loss_model1 = self.loss_fn_raw(model1_pred_logits, y_noisy[idx])
        loss_model2 = self.loss_fn_raw(model2_pred_logits, y_noisy[idx])

        # ema of loss
        loss_model1, loss_model2 = self.ema_loss(epoch, loss_model1, loss_model2, idx)
        loss_model1 = loss_model1.cuda()
        loss_model2 = loss_model2.cuda()
        
        # warmup by selecting all samples
        if epoch < self.args.epochs_warmup:  #self.warmup: #epoch < self.args.epochs_warmup:
            model1_select_idx = torch.arange(0, len(idx)).cuda()  # select all data
            model2_select_idx = torch.arange(0, len(idx)).cuda()  

            model1_agree_idx = model1_select_idx
            model2_agree_idx = model2_select_idx
            model1_disagree_idx = model1_select_idx
            model2_disagree_idx = model2_select_idx
        
        # Do sample selection
        else:
            alpha = self.args.correction_threshold_inlier  # tradeoff between loss and similarity score
            if self.args.experiment != "disagree_only":

                ###### Agreement module ######
                # JS divergence: smaller value has more similar distributions
                m = (model1_pred_logits.softmax(1) + model2_pred_logits.softmax(1))/2
                kl_main = F.kl_div(model1_pred_logits.log_softmax(1), m, reduction='none')
                kl_weight = F.kl_div(model2_pred_logits.log_softmax(1), m, reduction='none')
                similarity_score = 0.5 * torch.sum(kl_main, 1) + 0.5 * torch.sum(kl_weight, 1)

                # EMA of similarity score
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
                    sorted_scores, _ = torch.sort(sampling_score[combined_inlier_idx].view(-1))
                    scores_trimmed_inliers = sorted_scores[num_trim:-num_trim]
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
                    sorted_scores, _ = torch.sort(sampling_score[combined_outlier_idx].view(-1))
                    scores_trimmed_outliers = sorted_scores[num_trim:-num_trim]
                else:
                    scores_trimmed_outliers = sampling_score[combined_outlier_idx]

                
                # adaptive threshold for how many samples to select as clean
                sample_score_agreement_inliers_threshold = self.get_adaptive_threshold_agreement(scores_trimmed_inliers, noise_rate_inlier, sampling_score[combined_inlier_idx])
                sample_score_agreement_outliers_threshold = self.get_adaptive_threshold_agreement(scores_trimmed_outliers, noise_rate_outlier, sampling_score[combined_outlier_idx])
            
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
            
            if self.args.experiment != "agree_only":
                ###### Disagreement module ######
                
                if self.args.experiment == "disagree_only":
                    model1_select_idx = torch.empty(0, dtype=torch.long).cuda()
                    model2_select_idx = torch.empty(0, dtype=torch.long).cuda()
                
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
                    sorted_scores, _ = torch.sort(large_conf_small_loss_score_model1_inlier.view(-1))
                    scores_trimmed_inliers_model1 = sorted_scores[num_trim:-num_trim]
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
                    sorted_scores, _ = torch.sort(large_conf_small_loss_score_model1_outlier.view(-1))
                    scores_trimmed_outliers_model1 = sorted_scores[num_trim:-num_trim]
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
                    sorted_scores, _ = torch.sort(large_conf_small_loss_score_model2_inlier.view(-1))
                    scores_trimmed_inliers_model2 = sorted_scores[num_trim:-num_trim]
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
                    sorted_scores, _ = torch.sort(large_conf_small_loss_score_model2_outlier.view(-1))
                    scores_trimmed_outliers_model2 = sorted_scores[num_trim:-num_trim]
                else:
                    scores_trimmed_outliers_model2 = large_conf_small_loss_score_model2_outlier
            
                # adaptive threshold
                large_conf_sample_score_model1_inlier_threshold = self.get_adaptive_threshold_disagreement(scores_trimmed_inliers_model1, noise_rate_inlier, large_conf_small_loss_score_model1_inlier)
                large_conf_sample_score_model1_outlier_threshold = self.get_adaptive_threshold_disagreement(scores_trimmed_outliers_model1, noise_rate_outlier, large_conf_small_loss_score_model1_outlier)
                large_conf_sample_score_model2_inlier_threshold = self.get_adaptive_threshold_disagreement(scores_trimmed_inliers_model2, noise_rate_inlier, large_conf_small_loss_score_model2_inlier)
                large_conf_sample_score_model2_outlier_threshold = self.get_adaptive_threshold_disagreement(scores_trimmed_outliers_model2, noise_rate_outlier, large_conf_small_loss_score_model2_outlier)

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

        return model1_select_idx, model2_select_idx, model1_agree_idx, model2_agree_idx, model1_disagree_idx, model2_disagree_idx
        

    def update_labels_contrastive(self, model1_preds, model2_preds, y_noisy, idx, model1_selected_clean_idx, model2_selected_clean_idx, data, epoch, model1_agree_idx, gt_labels, model1_pred_logits, model2_pred_logits):
        '''
        Use the samples with clean labels from agree and disagree to train a contrastive learning model
        Then use the contrastive model to correct the labels of the remaining non-selected samples
        '''
        # concatenate the selected samples from model 1 and model 2
        # TODO: figure out how to use disagree samples: Currently only using samples from model1

        # create pairs: (inlier, inlier) is positive pair, (inlier, outlier) is negative pair
        
        #pred_inliers_idx = np.nonzero(gt_labels[model1_selected_clean_idx] == 0).flatten()
        #pred_outliers_idx = np.nonzero(gt_labels[model1_selected_clean_idx] == 1).flatten()
        
        # loss: Samples with large loss should be prioritized
        loss_model1 = self.loss_fn_raw(model1_pred_logits, y_noisy[idx])
        loss_model2 = self.loss_fn_raw(model2_pred_logits, y_noisy[idx])

        
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
        
        # Prepare positive pairs (inlier, inlier)
        positive_pairs = []
        for i in range(len(pred_inliers_idx)):
            # randomly sample another inlier except the current i
            sampled_indices = torch.cat((pred_inliers_idx[:i], pred_inliers_idx[i+1:]))[torch.randint(0, torch.cat((pred_inliers_idx[:i], pred_inliers_idx[i+1:])).size(0), (1,))] 
            positive_pairs.extend([(data[pred_inliers_idx][i], data[j]) for j in sampled_indices])
        
        # Prepare negative pairs (inlier, outlier)
        negative_pairs = []
        for i in range(len(pred_inliers_idx)):
            # randomly sample an outlier
            sampled_indices = pred_outliers_idx[torch.randint(0, pred_outliers_idx.size(0), (1,))]
            negative_pairs.extend([(data[pred_inliers_idx][i], data[j]) for j in sampled_indices])

        # Combine positive and negative pairs
        pairs = positive_pairs + negative_pairs
        labels = torch.cat((torch.zeros(len(positive_pairs)), torch.ones(len(negative_pairs))))  # labels: pos pairs are 0 and neg pairs are 1

        # Create DataLoader
        train_dataset = CustomDatasetCont(pairs, labels)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # train contrastive model
        for epoch in range(1):  # train contrastive model for 5 epochs
            for batch_idx, (pair, label) in enumerate(train_loader):
                self.optimizer_contrastive.zero_grad()
                output1, output2 = self.contrastive_model(pair[0].float().cuda(), pair[1].float().cuda())
                loss = self.contrastive_loss(output1, output2, label.cuda())
                loss.backward()
                self.optimizer_contrastive.step()

            # inference
            # output1, output2 = self.contrastive_model(data[pred_inliers_idx].float().cuda(), data[pred_inliers_idx].float().cuda())
            # training_inlier_embedding = output1.detach().cpu().numpy()  # only get the embedding of the inliers
            # centroid = np.mean(training_inlier_embedding, axis=0)  # calculate the centroid of the selected inliers

            # output1, output2 = self.contrastive_model(data[non_selected_1_idx].float().cuda(), data[non_selected_1_idx].float().cuda())  # get new embedding
            # non_selected_dists = np.linalg.norm(output1.detach().cpu().numpy() - centroid, axis=1)  # calculate the distance of each non-selected sample from the centroid

            # # get number of outliers
            # noisy_outlier_ratio = torch.argmax(y_noisy, dim=1).to(torch.float64).mean().item()
            # num_outliers = math.ceil(len(output1) * noisy_outlier_ratio * self.args.threshold)
            # num_inliers = math.ceil(len(output1) * (1-noisy_outlier_ratio) * self.args.threshold)

            # # sort the distances
            # sorted_outputs_idx = np.argsort(non_selected_dists)  # increasing order
            # outlier_idx = sorted_outputs_idx[-num_outliers:]
            # inlier_idx = sorted_outputs_idx[:num_inliers]

            # outlier_idx = non_selected_1_idx[outlier_idx]
            # inlier_idx = non_selected_1_idx[inlier_idx]
            
            # # correct the labels
            # model1_preds_old = model1_preds.clone()
            # model2_preds_old = model2_preds.clone()
            
            # model1_preds[outlier_idx] = 1
            # model1_preds[inlier_idx] = 0
            # model2_preds[outlier_idx] = 1
            # model2_preds[inlier_idx] = 0

            #print(f"F1 Score before contrastive learning epoch {epoch}: {f1_score(gt_labels[non_selected_1_idx].cpu(), model1_preds_old[non_selected_1_idx].cpu())}")
            #print(f"F1 Score of contrastive learning epoch {epoch}: {f1_score(gt_labels[non_selected_1_idx].cpu(), model1_preds[non_selected_1_idx].cpu())}")
            #print(" ")



        # get selected clean inlier embeddings to calculate the centroid
        output1, output2 = self.contrastive_model(data[pred_inliers_idx].float().cuda(), data[pred_inliers_idx].float().cuda())
        training_inlier_embedding = output1.detach().cpu().numpy()  # only get the embedding of the inliers
        centroid = np.mean(training_inlier_embedding, axis=0)  # calculate the centroid of the selected inliers

        # pass non-selected samples through the contrastive model
        output1_1, output2_1 = self.contrastive_model(data[non_selected_1_idx].float().cuda(), data[non_selected_1_idx].float().cuda())  # get new embedding
        non_selected_dists_1 = np.linalg.norm(output1_1.detach().cpu().numpy() - centroid, axis=1)  # calculate the distance of each non-selected sample from the centroid

        output1_1, output2_1 = self.contrastive_model(data[non_selected_2_idx].float().cuda(), data[non_selected_2_idx].float().cuda())  # get new embedding
        non_selected_dists_2 = np.linalg.norm(output1_1.detach().cpu().numpy() - centroid, axis=1)  # calculate the distance of each non-selected sample from the centroid

        # get number of outliers
        noisy_outlier_ratio = torch.argmax(y_noisy, dim=1).to(torch.float64).mean().item()
        #num_outliers_1 = math.ceil(len(output1) * noisy_outlier_ratio * self.args.threshold)
        #num_inliers_1 = math.ceil(len(output1) * (1-noisy_outlier_ratio) * self.args.threshold)

        #num_outliers_2 = math.ceil(len(output2) * noisy_outlier_ratio * self.args.threshold)
        #num_inliers_2 = math.ceil(len(output2) * (1-noisy_outlier_ratio) * self.args.threshold)

        num_outliers_1 = math.ceil(len(non_selected_dists_1) * noisy_outlier_ratio)
        num_inliers_1 = len(non_selected_dists_1) - num_outliers_1
        num_outliers_2 = math.ceil(len(non_selected_dists_2) * noisy_outlier_ratio)
        num_inliers_2 = len(non_selected_dists_2) - num_outliers_2

        # sort the distances
        sorted_outputs_idx_1 = np.argsort(non_selected_dists_1)  # increasing order
        outlier_idx_1 = sorted_outputs_idx_1[-num_outliers_1:]
        inlier_idx_1 = sorted_outputs_idx_1[:num_inliers_1]

        sorted_outputs_idx_2 = np.argsort(non_selected_dists_2)  # increasing order
        outlier_idx_2 = sorted_outputs_idx_2[-num_outliers_2:]
        inlier_idx_2 = sorted_outputs_idx_2[:num_inliers_2]

        # use adaptive threshold to select samples to correct 
        outlier_dist_1 = non_selected_dists_1[outlier_idx_1]
        inlier_dist_1 = non_selected_dists_1[inlier_idx_1]
        outlier_dist_2 = non_selected_dists_2[outlier_idx_2]
        inlier_dist_2 = non_selected_dists_2[inlier_idx_2]

        # remove the outliers and calculate adaptive threshold         
        if len(outlier_dist_1) > 1:
            outlier_dist_1 = np.clip(outlier_dist_1, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            outlier_dist_1 = power_transformer.fit_transform(outlier_dist_1.reshape(-1,1)).astype(np.float32).flatten()
        else:
            outlier_dist_1 = outlier_dist_1
        # remove the outliers
        if len(outlier_dist_1) > 2:
            # trim the outliers 
            num_trim = math.ceil(len(outlier_dist_1) * .05)
            sorted_scores = np.sort(outlier_dist_1)
            scores_trimmed_outliers_model1 = sorted_scores[num_trim:-num_trim]
        else:
            scores_trimmed_outliers_model1 = outlier_dist_1

        if len(inlier_dist_1) > 1:
            inlier_dist_1 = np.clip(inlier_dist_1, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            inlier_dist_1 = power_transformer.fit_transform(inlier_dist_1.reshape(-1,1)).astype(np.float32).flatten()
        else:
            inlier_dist_1 = inlier_dist_1
        # remove the outliers
        if len(inlier_dist_1) > 2:
            # trim the outliers 
            num_trim = math.ceil(len(inlier_dist_1) * .05)
            sorted_scores = np.sort(inlier_dist_1)
            scores_trimmed_inliers_model1 = sorted_scores[num_trim:-num_trim]
        else:
            scores_trimmed_inliers_model1 = inlier_dist_1

        if len(outlier_dist_2) > 1:
            outlier_dist_2 = np.clip(outlier_dist_2, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            outlier_dist_2 = power_transformer.fit_transform(outlier_dist_2.reshape(-1,1)).astype(np.float32).flatten()
        else:
            outlier_dist_2 = outlier_dist_2
        # remove the outliers
        if len(outlier_dist_2) > 2:
            # trim the outliers 
            num_trim = math.ceil(len(outlier_dist_2) * .05)
            sorted_scores = np.sort(outlier_dist_2)
            scores_trimmed_outliers_model2 = sorted_scores[num_trim:-num_trim]
        else:
            scores_trimmed_outliers_model2 = outlier_dist_2
        
        if len(inlier_dist_2) > 1:
            inlier_dist_2 = np.clip(inlier_dist_2, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            inlier_dist_2 = power_transformer.fit_transform(inlier_dist_2.reshape(-1,1)).astype(np.float32).flatten()
        else:
            inlier_dist_2 = inlier_dist_2
        # remove the outliers
        if len(inlier_dist_2) > 2:
            # trim the outliers 
            num_trim = math.ceil(len(inlier_dist_2) * .05)
            sorted_scores = np.sort(inlier_dist_2)
            scores_trimmed_inliers_model2 = sorted_scores[num_trim:-num_trim]
        else:
            scores_trimmed_inliers_model2 = inlier_dist_2         
        
        outlier_threshold_1 = scores_trimmed_outliers_model1.mean() + scores_trimmed_outliers_model1.std()
        inlier_threshold_1 = scores_trimmed_inliers_model1.mean() - scores_trimmed_inliers_model1.std()
        outlier_threshold_2 = scores_trimmed_outliers_model2.mean() + scores_trimmed_outliers_model2.std()
        inlier_threshold_2 = scores_trimmed_inliers_model2.mean() - scores_trimmed_inliers_model2.std()

        # get indexes of samples that comply with threshold 
        select_outlier_idx_1 = np.nonzero(outlier_dist_1 >= outlier_threshold_1)[0]
        select_inlier_idx_1 = np.nonzero(inlier_dist_1 <= inlier_threshold_1)[0]
        select_outlier_idx_2 = np.nonzero(outlier_dist_2 >= outlier_threshold_2)[0]
        select_inlier_idx_2 = np.nonzero(inlier_dist_2 <= inlier_threshold_2)[0]

        outlier_idx_1 = non_selected_1_idx[outlier_idx_1[select_outlier_idx_1]]
        inlier_idx_1 = non_selected_1_idx[inlier_idx_1[select_inlier_idx_1]]
        outlier_idx_2 = non_selected_2_idx[outlier_idx_2[select_outlier_idx_2]]
        inlier_idx_2 = non_selected_2_idx[inlier_idx_2[select_inlier_idx_2]]

        # correct the labels
        model1_preds_old = model1_preds.clone()
        model2_preds_old = model2_preds.clone()
        
        model1_preds[outlier_idx_1] = 1
        model1_preds[inlier_idx_1] = 0
        model2_preds[outlier_idx_2] = 1
        model2_preds[inlier_idx_2] = 0

        # print(f"F1 Score of selected clean labels: {f1_score(gt_labels[model1_selected_clean_idx].cpu(), model1_preds[model1_selected_clean_idx].cpu())}")
        print(f"F1 Score of contrastive learning model 1: {f1_score(gt_labels[np.concatenate((outlier_idx_1, inlier_idx_1))].cpu(), model1_preds[np.concatenate((outlier_idx_1, inlier_idx_1))].cpu())}")
        print(f"F1 Score of contrastive learning model 2: {f1_score(gt_labels[np.concatenate((outlier_idx_2, inlier_idx_2))].cpu(), model2_preds[np.concatenate((outlier_idx_2, inlier_idx_2))].cpu())}")
        # print(f"F1 Score of model preds: {f1_score(gt_labels[np.concatenate((outlier_indices, inlier_indices))].cpu(), model1_preds_old[np.concatenate((outlier_indices, inlier_indices))].cpu())}")
        # print(" ")
        corrected_idx_1 = np.concatenate((outlier_idx_1, inlier_idx_1))
        corrected_idx_2 = np.concatenate((outlier_idx_2, inlier_idx_2))
        model1_select_idx = torch.cat((model1_selected_clean_idx, torch.tensor(corrected_idx_1).cuda()))
        model2_select_idx = torch.cat((model2_selected_clean_idx, torch.tensor(corrected_idx_2).cuda()))
        model1_select_idx = torch.unique(model1_select_idx)
        model2_select_idx = torch.unique(model2_select_idx)
        
        return model1_select_idx, model2_select_idx, model1_preds, model2_preds, corrected_idx_1, gt_labels[np.concatenate((outlier_idx_1, inlier_idx_2))]
    

    def unity_sgd(self, dataloader, y_noisy, epoch, outlier_score, noise_rate_outlier, noise_rate_inlier):
        '''
        gradient descent for co-teaching

        Parameters
        ----------
        dataloader: data loader
        y_noisy : initial noisy pseudo labels
        epoch: current epoch

        Returns
        -------
        y_noisy: updated noisy pseudo labels
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
        model1_correction_centroid_pred = []
        model2_correction_centroid_pred = []
        model1_correction_grad_pred = []
        model2_correction_grad_pred = []
        model1_correction_centroid_gt = []
        model2_correction_centroid_gt = []
        model1_correction_grad_gt = []
        model2_correction_grad_gt = []
        contrastive_pred = []
        contrastive_gt = []

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
            
            else:
                model1_pred_logits = self.model1(data)
                model1_pred_prob = torch.softmax(model1_pred_logits, dim=1)
                model1_outlier_scores = model1_pred_prob[:,1]
                noisy_outlier_ratio = torch.argmax(y_noisy, dim=1).to(torch.float64).mean().item()
                num_outliers = math.ceil(noisy_outlier_ratio * len(idx))
                
                #gt_outlier_rate = label.to(torch.float64).mean().item()
                #num_outliers = math.ceil(gt_outlier_rate * len(idx))
                
                conf_sorted = torch.argsort(model1_outlier_scores)
                pred_outlier = conf_sorted[-num_outliers:]
                model1_pred_label = torch.zeros(len(idx)).cuda().to(int)
                model1_pred_label[pred_outlier] = 1

                #model1_pred_label = torch.argmax(model1_pred_prob, dim=1)

                # pass data through model2 for gradient calculation
                model2_pred_logits = self.model2(data)
                model2_pred_prob = torch.softmax(model2_pred_logits, dim=1)
                model2_outlier_scores = model2_pred_prob[:,1]
                noisy_outlier_ratio = torch.argmax(y_noisy, dim=1).to(torch.float64).mean().item()
                num_outliers = math.ceil(noisy_outlier_ratio * len(idx))

                #gt_outlier_rate = label.to(torch.float64).mean().item()
                #num_outliers = math.ceil(gt_outlier_rate * len(idx))

                conf_sorted = torch.argsort(model2_outlier_scores)
                pred_outlier = conf_sorted[-num_outliers:]
                model2_pred_label = torch.zeros(len(idx)).cuda().to(int)
                model2_pred_label[pred_outlier] = 1   

                #model2_pred_label = torch.argmax(model2_pred_prob, dim=1)
                
                # Sample selection
                model1_select_idx, model2_select_idx, model1_agree_idx, model2_agree_idx, model1_disagree_idx, model2_disagree_idx = self.sample_selection(model1_pred_logits, model2_pred_logits, idx, epoch, y_noisy, model1_pred_label, model2_pred_label, noise_rate_outlier, noise_rate_inlier)

                model1_selected_preds_clean[idx[model1_select_idx.cpu()]] = model1_pred_label[model1_select_idx].cpu()
                model2_selected_preds_clean[idx[model2_select_idx.cpu()]] = model2_pred_label[model2_select_idx].cpu()
                model1_selected_clean.append(idx[model1_select_idx.cpu()])
                model2_selected_clean.append(idx[model2_select_idx.cpu()])            
                
                # track performance of each module
                model1_agreement_pred.append(model1_pred_label[model1_agree_idx])
                model2_agreement_pred.append(model2_pred_label[model2_agree_idx])
                model1_disagreement_pred.append(model1_pred_label[model1_disagree_idx])
                model2_disagreement_pred.append(model2_pred_label[model2_disagree_idx])

                model1_agreement_gt.append(label[model1_agree_idx])
                model2_agreement_gt.append(label[model2_agree_idx])
                model1_disagreement_gt.append(label[model1_disagree_idx])
                model2_disagreement_gt.append(label[model2_disagree_idx])

            # Label Correction               
            if (self.args.experiment == "unity" or self.args.experiment == "synthetic" or  self.args.experiment == "refurbishment_only" or self.args.experiment == "no_threshold"):
                if epoch >= self.args.epochs_warmup and self.args.correction_method == "contrastive":  # not self.warmup and self.args.correction_method == "contrastive": #epoch >= self.args.epochs_warmup and self.args.correction_method == "contrastive":
                    model1_pred_label_before_correction = model1_pred_label.clone()
                    model2_pred_label_before_correction = model2_pred_label.clone()
                
                    model1_select_idx, model2_select_idx, model1_pred_label, model2_pred_label, corrected_contrastive_idx, gt_contrastive = self.update_labels_contrastive(model1_pred_label, model2_pred_label, y_noisy, idx, model1_select_idx, model2_select_idx, data, epoch, model1_agree_idx, label, model1_pred_logits, model2_pred_logits)
                    
                    # track accuracy of contrastive correction
                    contrastive_pred.append(model1_pred_label[corrected_contrastive_idx])
                    contrastive_gt.append(gt_contrastive)
                
                elif epoch >= self.args.epochs_warmup and self.args.correction_method == "contrastive_highloss":  # not self.warmup and self.args.correction_method == "contrastive_highloss": # epoch >= self.args.epochs_warmup and self.args.correction_method == "contrastive_highloss": 
                    model1_pred_label_before_correction = model1_pred_label.clone()
                    model2_pred_label_before_correction = model2_pred_label.clone()

                    model1_select_idx, model2_select_idx, model1_pred_label, model2_pred_label, corrected_contrastive_idx, gt_contrastive = self.update_labels_contrastive_highloss(model1_pred_label, model2_pred_label, y_noisy, idx, model1_select_idx, model2_select_idx, data, epoch, model1_agree_idx, label, model1_pred_logits, model2_pred_logits)
                
                    # track accuracy of contrastive correction
                    contrastive_pred.append(model1_pred_label[corrected_contrastive_idx])
                    contrastive_gt.append(gt_contrastive)

                elif epoch >= self.args.epochs_warmup and (self.args.correction_method == "contrastive_all"):

                    model1_select_idx, model2_select_idx, model1_pred_label, model2_pred_label, corrected_contrastive_idx, gt_contrastive = self.update_labels_all(model1_pred_label, model2_pred_label, y_noisy, idx, model1_select_idx, model2_select_idx, data, epoch, model1_agree_idx, label, model1_pred_logits, model2_pred_logits)
                    
                    # track accuracy of contrastive correction
                    contrastive_pred.append(model1_pred_label[corrected_contrastive_idx])
                    contrastive_gt.append(gt_contrastive)
                
                elif epoch >= self.args.epochs_warmup and self.args.correction_method == "custom_CTL":
                    model1_select_idx, model2_select_idx, model1_pred_label, model2_pred_label, corrected_contrastive_idx, gt_contrastive = self.update_labels_custom_ctr(model1_pred_label, model2_pred_label, y_noisy, idx, model1_select_idx, model2_select_idx, data, epoch, model1_agree_idx, label, model1_pred_logits, model2_pred_logits)

                
                else:
                    model1_correction_centroid_pred.append(model1_pred_label[model1_select_idx])
                    model2_correction_centroid_pred.append(model2_pred_label[model2_select_idx])
                    model1_correction_grad_pred.append(model1_pred_label[model1_select_idx])
                    model2_correction_grad_pred.append(model2_pred_label[model2_select_idx])
                    model1_correction_centroid_gt.append(label[model1_select_idx])
                    model2_correction_centroid_gt.append(label[model2_select_idx])
                    model1_correction_grad_gt.append(label[model1_select_idx])
                    model2_correction_grad_gt.append(label[model2_select_idx])
                    contrastive_pred.append(model1_pred_label[model1_select_idx])
                    contrastive_gt.append(label[model1_select_idx])
        
             
            if epoch < self.args.epochs_warmup:
                model1_cf_data, model1_cf_label = data[model1_select_idx], torch.argmax(y_noisy[idx], axis=1)[model1_select_idx]
                model2_cf_data, model2_cf_label = data[model2_select_idx], torch.argmax(y_noisy[idx], axis=1)[model2_select_idx]

            else:  # use model predictions
                # shuffle the selected indices
                model1_select_idx = model1_select_idx[torch.randperm(len(model1_select_idx))]
                model2_select_idx = model2_select_idx[torch.randperm(len(model2_select_idx))]
                
                if self.args.experiment != "unity" and self.args.experiment != "synthetic":
                    # do only selection: No refurbishment
                    model1_cf_data, model1_cf_label = data[model1_select_idx], torch.argmax(y_noisy[idx], axis=1)[model1_select_idx]  
                    model2_cf_data, model2_cf_label = data[model2_select_idx], torch.argmax(y_noisy[idx], axis=1)[model2_select_idx]

                else:
                    model1_cf_data, model1_cf_label = data[model1_select_idx], model1_pred_label[model1_select_idx]  
                    model2_cf_data, model2_cf_label = data[model2_select_idx], model2_pred_label[model2_select_idx]
            
            if self.args.experiment != "refurbishment_only":
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

        return y_noisy, model1_agreement_pred, model2_agreement_pred, model1_agreement_gt, model2_agreement_gt, model1_disagreement_pred, model2_disagreement_pred, model1_disagreement_gt, model2_disagreement_gt, model1_selected_clean, model2_selected_clean, model1_selected_preds_clean


    def train_unity(self, dataloader, y_noisy, outlier_score, noise_rate_outlier, noise_rate_inlier):
        '''
        Main training loop for Unity
        
        Parameters
        ----------
        dataloader: data loader
        y_noisy : initial noisy pseudo labels 

        Returns
        -------
        Nothing: Will print performance metrics of Unity
        '''        
        # Store history of losses and confidences
        self.model1_ema = EMARecorder(y_noisy)
        self.model2_ema = EMARecorder(y_noisy)
        self.initial_label_constant = y_noisy.clone()
       
        for epoch in range(self.args.epochs):
            print(" ")
            print(f"####### Epoch {epoch} #######")
            self.model1.train()
            self.model2.train()

            # SGD
            y_noisy_updated, model1_agreement_pred, model2_agreement_pred, model1_agreement_gt, model2_agreement_gt, model1_disagreement_pred, model2_disagreement_pred, model1_disagreement_gt, model2_disagreement_gt, model1_selected_clean, model2_selected_clean, model1_selected_preds_clean = self.unity_sgd(dataloader, y_noisy, epoch, outlier_score, noise_rate_outlier, noise_rate_inlier)
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
                pred_labels_ctl, label_ctl = self.inference_ctl(dataloader, y_noisy, model1_selected_clean, model1_selected_preds_clean)

                f1_score_ctl = f1_score(label_ctl.cpu(), pred_labels_ctl)
                print(f"F1 score of ctl model: {f1_score_ctl}")
                                    
                
                
        print("Done training Unity!")
        # write results to csv
        csv_file_path = f"results/results_final_{self.args.experiment}.csv"
        path_exists = False
        if os.path.exists(csv_file_path):
            path_exists = True
        
        if self.args.experiment != "refurbishment_only":
            with open(csv_file_path, mode='a', newline='') as file:
                csv_writer = csv.writer(file)
                if not path_exists:
                    csv_writer.writerow(["Method", "Dataset", "seed", "batch_norm", "f1", "precision", "recall", "auc"])
                csv_writer.writerow([self.args.method, self.args.dataset, self.args.seed, self.args.batch_norm, model1_f1, model1_precision, model1_recall, model1_auc])

        else:
            with open(csv_file_path, mode='a', newline='') as file:
                csv_writer = csv.writer(file)
                if not path_exists:
                    csv_writer.writerow(["Method", "Dataset", "seed", "batch_norm", "f1"])
                csv_writer.writerow([self.args.method, self.args.dataset, self.args.seed, self.args.batch_norm, f1_score_ctl])
            


    # code to use centroid method to get new labels for high loss samples 
    def update_labels_contrastive_highloss(self, model1_preds, model2_preds, y_noisy, idx, model1_selected_clean_idx, model2_selected_clean_idx, data, epoch, model1_agree_idx, gt_labels, model1_pred_logits, model2_pred_logits):
        '''
        Use the samples with clean labels from agree and disagree to train a contrastive learning model
        Then use the contrastive model to correct the labels of the remaining non-selected samples
        '''
        # concatenate the selected samples from model 1 and model 2
        # TODO: figure out how to use disagree samples: Currently only using samples from model1

        # create pairs: (inlier, inlier) is positive pair, (inlier, outlier) is negative pair
        
        #pred_inliers_idx = np.nonzero(gt_labels[model1_selected_clean_idx] == 0).flatten()
        #pred_outliers_idx = np.nonzero(gt_labels[model1_selected_clean_idx] == 1).flatten()

        # loss: Samples with large loss should be prioritized
        loss_model1 = self.loss_fn_raw(model1_pred_logits, y_noisy[idx])
        loss_model2 = self.loss_fn_raw(model2_pred_logits, y_noisy[idx])

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
        
        loss_model1_non_selected = loss_model1[non_selected_1_idx]
        loss_model2_non_selected = loss_model2[non_selected_2_idx]

        # TODO: prob need to split the non-selected samples into inliers and outliers
        # sort the indexes of the non-selected samples by loss
        sorted_loss_model1_idx = torch.argsort(loss_model1_non_selected)  # increasing order
        sorted_loss_model2_idx = torch.argsort(loss_model2_non_selected)  # increasing order
        
        # Prepare positive pairs (inlier, inlier)
        positive_pairs = []
        for i in range(len(pred_inliers_idx)):
            # randomly sample another inlier except the current i
            sampled_indices = torch.cat((pred_inliers_idx[:i], pred_inliers_idx[i+1:]))[torch.randint(0, torch.cat((pred_inliers_idx[:i], pred_inliers_idx[i+1:])).size(0), (1,))] 
            positive_pairs.extend([(data[pred_inliers_idx][i], data[j]) for j in sampled_indices])
        
        # Prepare negative pairs (inlier, outlier)
        negative_pairs = []
        for i in range(len(pred_inliers_idx)):
            # randomly sample an outlier
            sampled_indices = pred_outliers_idx[torch.randint(0, pred_outliers_idx.size(0), (1,))]
            negative_pairs.extend([(data[pred_inliers_idx][i], data[j]) for j in sampled_indices])

        # Combine positive and negative pairs
        pairs = positive_pairs + negative_pairs
        labels = torch.cat((torch.zeros(len(positive_pairs)), torch.ones(len(negative_pairs))))  # labels: pos pairs are 0 and neg pairs are 1

        # Create DataLoader
        train_dataset = CustomDatasetCont(pairs, labels)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # train contrastive model
        for epoch in range(1):  # train contrastive model for 5 epochs
            for batch_idx, (pair, label) in enumerate(train_loader):
                self.optimizer_contrastive.zero_grad()
                output1, output2 = self.contrastive_model(pair[0].float().cuda(), pair[1].float().cuda())
                loss = self.contrastive_loss(output1, output2, label.cuda())
                loss.backward()
                self.optimizer_contrastive.step()

        # get selected clean inlier embeddings to calculate the centroid
        output1, output2 = self.contrastive_model(data[pred_inliers_idx].float().cuda(), data[pred_inliers_idx].float().cuda())
        training_inlier_embedding = output1.detach().cpu().numpy()  # only get the embedding of the inliers
        centroid = np.mean(training_inlier_embedding, axis=0)  # calculate the centroid of the selected inliers

        # pass non-selected samples through the contrastive model
        output1_1, output2_1 = self.contrastive_model(data[non_selected_1_idx].float().cuda(), data[non_selected_1_idx].float().cuda())  # get new embedding
        non_selected_dists_1 = np.linalg.norm(output1_1.detach().cpu().numpy() - centroid, axis=1)  # calculate the distance of each non-selected sample from the centroid

        output1_2, output2_2 = self.contrastive_model(data[non_selected_2_idx].float().cuda(), data[non_selected_2_idx].float().cuda())  # get new embedding
        non_selected_dists_2 = np.linalg.norm(output1_2.detach().cpu().numpy() - centroid, axis=1)

        # get number of outliers
        noisy_outlier_ratio = torch.argmax(y_noisy, dim=1).to(torch.float64).mean().item()
        num_outliers = math.ceil(len(output1) * noisy_outlier_ratio)
        #num_inliers = math.ceil(len(output1) * (1-noisy_outlier_ratio))
        num_inliers = len(output1) - num_outliers
        
        # sort the distances
        sorted_outputs_idx_1 = np.argsort(non_selected_dists_1)  # increasing order
        outlier_idx_1 = sorted_outputs_idx_1[-num_outliers:]
        inlier_idx_1 = sorted_outputs_idx_1[:num_inliers]
        outlier_idx_1 = non_selected_1_idx[outlier_idx_1]
        inlier_idx_1 = non_selected_1_idx[inlier_idx_1]

        sorted_outputs_idx_2 = np.argsort(non_selected_dists_2)  # increasing order
        outlier_idx_2 = sorted_outputs_idx_2[-num_outliers:]
        inlier_idx_2 = sorted_outputs_idx_2[:num_inliers]
        outlier_idx_2 = non_selected_2_idx[outlier_idx_2]
        inlier_idx_2 = non_selected_2_idx[inlier_idx_2]

        # contrastive learning labels for all non-selected samples
        contrastive_pred_1 = model1_preds.clone()
        contrastive_pred_1[outlier_idx_1] = 1
        contrastive_pred_1[inlier_idx_1] = 0

        contrastive_pred_2 = model2_preds.clone()
        contrastive_pred_2[outlier_idx_2] = 1
        contrastive_pred_2[inlier_idx_2] = 0


        # select the samples with the largest losses, update their labels and include in model training  
        inlier_preds_model1_idx = np.nonzero(model1_preds[non_selected_1_idx] == 0).flatten()
        inlier_preds_model2_idx = np.nonzero(model2_preds[non_selected_2_idx] == 0).flatten()
        outlier_preds_model1_idx = np.nonzero(model1_preds[non_selected_1_idx] == 1).flatten()
        outlier_preds_model2_idx = np.nonzero(model2_preds[non_selected_2_idx] == 1).flatten()

        # adaptive threshold to select samples with large loss
        inlier_preds_model1_loss = loss_model1[inlier_preds_model1_idx]
        inlier_preds_model2_loss = loss_model2[inlier_preds_model2_idx]
        outlier_preds_model1_loss = loss_model1[outlier_preds_model1_idx]
        outlier_preds_model2_loss = loss_model2[outlier_preds_model2_idx]
        
        # remove the outliers and calculate adaptive threshold         
        if len(inlier_preds_model1_loss) > 1:
            inlier_preds_model1_loss = torch.clamp(inlier_preds_model1_loss, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            inlier_preds_model1_loss = torch.from_numpy(power_transformer.fit_transform(inlier_preds_model1_loss.detach().cpu().reshape(-1,1)).astype(np.float32)).flatten().cuda()
        else:
            inlier_preds_model1_loss = inlier_preds_model1_loss
        # remove the outliers
        if len(inlier_preds_model1_loss) > 2:
            # trim the outliers 
            num_trim = math.ceil(len(inlier_preds_model1_loss) * .05)
            sorted_scores = torch.sort(inlier_preds_model1_loss)[0]
            scores_trimmed_inliers_model1 = sorted_scores[num_trim:-num_trim]
        else:
            scores_trimmed_inliers_model1 = inlier_preds_model1_loss
        
        if len(inlier_preds_model2_loss) > 1:
            inlier_preds_model2_loss = torch.clamp(inlier_preds_model2_loss, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            inlier_preds_model2_loss = torch.from_numpy(power_transformer.fit_transform(inlier_preds_model2_loss.detach().cpu().reshape(-1,1)).astype(np.float32)).flatten().cuda()
        else:
            inlier_preds_model2_loss = inlier_preds_model2_loss
        # remove the outliers
        if len(inlier_preds_model2_loss) > 2:
            # trim the outliers 
            num_trim = math.ceil(len(inlier_preds_model2_loss) * .05)
            sorted_scores = torch.sort(inlier_preds_model2_loss)[0]
            scores_trimmed_inliers_model2 = sorted_scores[num_trim:-num_trim]
        else:
            scores_trimmed_inliers_model2 = inlier_preds_model2_loss

        if len(outlier_preds_model1_loss):
            outlier_preds_model1_loss = torch.clamp(outlier_preds_model1_loss, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            outlier_preds_model1_loss = torch.from_numpy(power_transformer.fit_transform(outlier_preds_model1_loss.detach().cpu().reshape(-1,1)).astype(np.float32)).flatten().cuda()
        else:
            outlier_preds_model1_loss = outlier_preds_model1_loss
        # remove the outliers
        if len(outlier_preds_model1_loss) > 2:
            # trim the outliers 
            num_trim = math.ceil(len(outlier_preds_model1_loss) * .05)
            sorted_scores = torch.sort(outlier_preds_model1_loss)[0]
            scores_trimmed_outliers_model1 = sorted_scores[num_trim:-num_trim]
        else:
            scores_trimmed_outliers_model1 = outlier_preds_model1_loss

        if len(outlier_preds_model2_loss):
            outlier_preds_model2_loss = torch.clamp(outlier_preds_model2_loss, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            outlier_preds_model2_loss = torch.from_numpy(power_transformer.fit_transform(outlier_preds_model2_loss.detach().cpu().reshape(-1,1)).astype(np.float32)).flatten().cuda()
        else:
            outlier_preds_model2_loss = outlier_preds_model2_loss
        # remove the outliers
        if len(outlier_preds_model2_loss) > 2:
            # trim the outliers 
            num_trim = math.ceil(len(outlier_preds_model2_loss) * .05)
            sorted_scores = torch.sort(outlier_preds_model2_loss)[0]
            scores_trimmed_outliers_model2 = sorted_scores[num_trim:-num_trim]
        else:
            scores_trimmed_outliers_model2 = outlier_preds_model2_loss

        # calculate the threshold
        threshold_inliers_model1 = torch.mean(scores_trimmed_inliers_model1) + torch.std(scores_trimmed_inliers_model1)
        threshold_inliers_model2 = torch.mean(scores_trimmed_inliers_model2) + torch.std(scores_trimmed_inliers_model2)
        threshold_outliers_model1 = torch.mean(scores_trimmed_outliers_model1) + torch.std(scores_trimmed_outliers_model1)
        threshold_outliers_model2 = torch.mean(scores_trimmed_outliers_model2) + torch.std(scores_trimmed_outliers_model2)

        # select the samples with large loss
        select_inlier_idx_1 = torch.nonzero(inlier_preds_model1_loss >= threshold_inliers_model1).squeeze()
        select_outlier_idx_1 = torch.nonzero(outlier_preds_model1_loss >= threshold_outliers_model1).squeeze()
        select_inlier_idx_2 = torch.nonzero(inlier_preds_model2_loss >= threshold_inliers_model2).squeeze()
        select_outlier_idx_2 = torch.nonzero(outlier_preds_model2_loss >= threshold_outliers_model2).squeeze()

        inlier_idx_1 = non_selected_1_idx[select_inlier_idx_1.cpu()]
        outlier_idx_1 = non_selected_1_idx[select_outlier_idx_1.cpu()]
        inlier_idx_2 = non_selected_2_idx[select_inlier_idx_2.cpu()]
        outlier_idx_2 = non_selected_2_idx[select_outlier_idx_2.cpu()]

        # correct the labels
        model1_preds_old = model1_preds.clone()
        model2_preds_old = model2_preds.clone()

        model1_corr_labels_idx = torch.cat((torch.tensor(inlier_idx_1), torch.tensor(outlier_idx_1))).cuda()
        model2_corr_labels_idx = torch.cat((torch.tensor(inlier_idx_2), torch.tensor(outlier_idx_2))).cuda()

        model1_preds[inlier_idx_1] = contrastive_pred_1[inlier_idx_1]
        model1_preds[outlier_idx_1] = contrastive_pred_1[outlier_idx_1]
        model2_preds[inlier_idx_2] = contrastive_pred_2[inlier_idx_2]
        model2_preds[outlier_idx_2] = contrastive_pred_2[outlier_idx_2]

        # print(f"F1 Score of selected clean labels: {f1_score(gt_labels[model1_selected_clean_idx].cpu(), model1_preds[model1_selected_clean_idx].cpu())}")
        print(f"F1 Score of contrastive learning model 1: {f1_score(gt_labels[model1_corr_labels_idx].cpu(), model1_preds[model1_corr_labels_idx].cpu())}")
        print(f"F1 Score of contrastive learning model 2: {f1_score(gt_labels[model2_corr_labels_idx].cpu(), model1_preds[model2_corr_labels_idx].cpu())}")

        # print(f"F1 Score of model preds: {f1_score(gt_labels[np.concatenate((outlier_indices, inlier_indices))].cpu(), model1_preds_old[np.concatenate((outlier_indices, inlier_indices))].cpu())}")
        # print(" ")
        # corrected_idx = np.concatenate((outlier_idx, inlier_idx))
        # model1_select_idx = torch.cat((model1_selected_clean_idx, torch.tensor(corrected_idx).cuda()))
        # model2_select_idx = torch.cat((model2_selected_clean_idx, torch.tensor(corrected_idx).cuda()))
        model1_select_idx = torch.cat((model1_selected_clean_idx, model1_corr_labels_idx))
        model2_select_idx = torch.cat((model2_selected_clean_idx,model2_corr_labels_idx))
        model1_select_idx = torch.unique(model1_select_idx)
        model2_select_idx = torch.unique(model2_select_idx)
        
        return model1_select_idx, model2_select_idx, model1_preds, model2_preds, model1_corr_labels_idx, gt_labels[model1_selected_clean_idx] #gt_labels[np.concatenate((outlier_idx, inlier_idx))]
    


    def update_labels_custom_ctr(self, model1_preds, model2_preds, y_noisy, idx, model1_selected_clean_idx, model2_selected_clean_idx, data, epoch, model1_agree_idx, gt_labels, model1_pred_logits, model2_pred_logits):
        # combine model 1 and model 2 selected as clean samples
        # TODO: we need to find a way to combine the clean samples from model 1 and model 2
        selected_clean_idx = torch.cat((model1_selected_clean_idx, model2_selected_clean_idx))
        selected_clean_idx = torch.unique(selected_clean_idx)
        
        self.sup_contrastive_loss.train()
        
        # index of selected as clean inliers 
        clean_inlier_idx = np.nonzero(model1_preds[model1_selected_clean_idx] == 0).flatten()
        
        # selected as clean samples
        selected_clean_data = data[model1_selected_clean_idx].cuda()
        selected_clean_labels = model1_preds[model1_selected_clean_idx].cuda()

        # non-selected samples 
        non_selected_1_idx = np.nonzero(~np.isin(np.arange(len(data)), model1_selected_clean_idx.cpu()))[0]
        non_selected_2_idx = np.nonzero(~np.isin(np.arange(len(data)), model2_selected_clean_idx.cpu()))[0]
        
        # get the embeddings of the selected clean samples
        embeddings = self.contrastive_model(selected_clean_data.float().cuda())

        # get loss
        ctr_loss = self.sup_contrastive_loss(embeddings, selected_clean_labels)

        # update CTR model 
        self.optimizer_contrastive.zero_grad()
        ctr_loss.backward()
        self.optimizer_contrastive.step()

        # get new labels for non-selected data
        self.sup_contrastive_loss.eval()

        # get embeddings of selected inliers
        clean_inliers_embeddings = embeddings[clean_inlier_idx]  # check this indexing 

        centroid = torch.mean(clean_inliers_embeddings, dim=0)  # calculate centroid of clean inlier embeddings

        # pass non-selected samples through the contrastive model
        nonselected_embeddings_1 = self.contrastive_model(data[non_selected_1_idx].float().cuda())
        nonselected_embedding_2 = self.contrastive_model(data[non_selected_2_idx].float().cuda())

        # calculate the distance of each non-selected sample from the centroid
        non_selected_dists_1 = torch.linalg.norm(nonselected_embeddings_1 - centroid, dim=1)
        non_selected_dists_2 = torch.linalg.norm(nonselected_embedding_2 - centroid, dim=1)

        # get number of outliers
        noisy_outlier_ratio = torch.argmax(y_noisy, dim=1).to(torch.float64).mean().item()
        num_outliers_1 = math.ceil(len(non_selected_dists_1) * noisy_outlier_ratio)
        num_inliers_1 = len(non_selected_dists_1) - num_outliers_1
        num_outliers_2 = math.ceil(len(non_selected_dists_2) * noisy_outlier_ratio)
        num_inliers_2 = len(non_selected_dists_2) - num_outliers_2
        
        # sort the distances
        sorted_outputs_idx_1 = torch.argsort(non_selected_dists_1)  # increasing order
        outlier_idx_1 = sorted_outputs_idx_1[-num_outliers_1:]
        inlier_idx_1 = sorted_outputs_idx_1[:num_inliers_1]
        outlier_idx_1 = non_selected_1_idx[outlier_idx_1.cpu()]
        inlier_idx_1 = non_selected_1_idx[inlier_idx_1.cpu()]

        sorted_outputs_idx_2 = torch.argsort(non_selected_dists_2)  # increasing order
        outlier_idx_2 = sorted_outputs_idx_2[-num_outliers_2:]
        inlier_idx_2 = sorted_outputs_idx_2[:num_inliers_2]
        outlier_idx_2 = non_selected_2_idx[outlier_idx_2.cpu()]
        inlier_idx_2 = non_selected_2_idx[inlier_idx_2.cpu()]
        

        # contrastive learning labels for all non-selected samples
        contrastive_pred_1 = model1_preds.clone()
        contrastive_pred_1[outlier_idx_1] = 1
        contrastive_pred_1[inlier_idx_1] = 0

        contrastive_pred_2 = model2_preds.clone()
        contrastive_pred_2[outlier_idx_2] = 1
        contrastive_pred_2[inlier_idx_2] = 0

        model1_corr_labels_idx = torch.cat((torch.tensor(inlier_idx_1), torch.tensor(outlier_idx_1))).cuda()
        model2_corr_labels_idx = torch.cat((torch.tensor(inlier_idx_2), torch.tensor(outlier_idx_2))).cuda()

        model1_preds = contrastive_pred_1
        model2_preds = contrastive_pred_2

        print(f"F1 Score of contrastive learning model 1: {f1_score(gt_labels[model1_corr_labels_idx].cpu(), model1_preds[model1_corr_labels_idx].cpu())}")
        print(f"F1 Score of contrastive learning model 2: {f1_score(gt_labels[model2_corr_labels_idx].cpu(), model2_preds[model2_corr_labels_idx].cpu())}")

        model1_select_idx = torch.cat((model1_selected_clean_idx, model1_corr_labels_idx))
        model2_select_idx = torch.cat((model2_selected_clean_idx, model2_corr_labels_idx))
        model1_select_idx = torch.unique(model1_select_idx)
        model2_select_idx = torch.unique(model2_select_idx)

        return model1_select_idx, model2_select_idx, model1_preds, model2_preds, model1_corr_labels_idx, gt_labels[model1_selected_clean_idx]



    def update_labels_all(self, model1_preds, model2_preds, y_noisy, idx, model1_selected_clean_idx, model2_selected_clean_idx, data, epoch, model1_agree_idx, gt_labels, model1_pred_logits, model2_pred_logits):
        '''
        Use the samples with clean labels from agree and disagree to train a contrastive learning model
        Then use the contrastive model to correct the labels of the remaining non-selected samples
        '''
        # concatenate the selected samples from model 1 and model 2
        # create pairs: (inlier, inlier) is positive pair, (inlier, outlier) is negative pair

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
        
        # Prepare positive pairs (inlier, inlier) and negative pairs (inlier, outlier)
        remaining_indices_inliers = list(range(pred_inliers_idx.shape[0]))
        remaining_indices_outlier = list(range(pred_outliers_idx.shape[0]))
        negative_pairs = []
        positive_pairs = []
        for i in range(len(pred_inliers_idx)):
            # randomly sample another inlier except the current i
            j = random.choice(remaining_indices_inliers[:i] + remaining_indices_inliers[i+1:])
            remaining_indices_inliers.remove(j)
            positive_pairs.append((data[pred_inliers_idx][i], data[pred_inliers_idx][j]))

            # randomly sample an outlier
            outlier_idx = random.choice(remaining_indices_outlier)
            remaining_indices_outlier.remove(outlier_idx)
            negative_pairs.append((data[pred_inliers_idx][i], data[pred_outliers_idx][outlier_idx]))
            
            # if outlier set is empty, add all samples back
            if len(remaining_indices_outlier) == 0:
                remaining_indices_outlier = list(range(pred_outliers_idx.shape[0]))
        
        # Combine positive and negative pairs
        pairs = positive_pairs + negative_pairs
        labels = torch.cat((torch.zeros(len(positive_pairs)), torch.ones(len(negative_pairs))))  # labels: pos pairs are 0 and neg pairs are 1

        # Create DataLoader
        train_dataset = CustomDatasetCont(pairs, labels)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        # train contrastive model
        self.contrastive_model.train()
        for epoch in range(1):
            for batch_idx, (pair, label) in enumerate(train_loader):
                self.optimizer_contrastive.zero_grad()
                output1, output2 = self.contrastive_model(pair[0].float().cuda(), pair[1].float().cuda())
                loss = self.contrastive_loss(output1, output2, label.cuda())
                loss.backward()
                self.optimizer_contrastive.step()

        self.contrastive_model.eval()
        # get selected clean inlier embeddings to calculate the centroid
        output1, output2 = self.contrastive_model(data[pred_inliers_idx].float().cuda(), data[pred_inliers_idx].float().cuda())
        training_inlier_embedding = output1.detach().cpu().numpy()  # only get the embedding of the inliers
        centroid = np.mean(training_inlier_embedding, axis=0)  # calculate the centroid of the selected inliers

        # pass non-selected samples through the contrastive model
        output1_1, output2_1 = self.contrastive_model(data[non_selected_1_idx].float().cuda(), data[non_selected_1_idx].float().cuda())  # get new embedding
        non_selected_dists_1 = np.linalg.norm(output1_1.detach().cpu().numpy() - centroid, axis=1)  # calculate the distance of each non-selected sample from the centroid

        output1_2, output2_2 = self.contrastive_model(data[non_selected_2_idx].float().cuda(), data[non_selected_2_idx].float().cuda())  # get new embedding
        non_selected_dists_2 = np.linalg.norm(output1_2.detach().cpu().numpy() - centroid, axis=1)

        # get number of outliers
        noisy_outlier_ratio = torch.argmax(y_noisy, dim=1).to(torch.float64).mean().item()
        num_outliers = math.ceil(len(output1) * noisy_outlier_ratio)
        #num_inliers = math.ceil(len(output1) * (1-noisy_outlier_ratio))
        num_inliers = len(output1) - num_outliers

        # number of samples to ignore
        num_ignore_outliers = 0 #math.ceil(num_outliers * .1)
        num_ignore_inliers = 0 #math.ceil(num_inliers * .1)
        
        # sort the distances
        sorted_outputs_idx_1 = np.argsort(non_selected_dists_1)  # increasing order
        outlier_idx_1 = sorted_outputs_idx_1[-(num_outliers-num_ignore_outliers):]
        inlier_idx_1 = sorted_outputs_idx_1[:(num_inliers-num_ignore_inliers)]
        outlier_idx_1 = non_selected_1_idx[outlier_idx_1]
        inlier_idx_1 = non_selected_1_idx[inlier_idx_1]

        # ignore the unknown samples

        sorted_outputs_idx_2 = np.argsort(non_selected_dists_2)  # increasing order
        outlier_idx_2 = sorted_outputs_idx_2[-(num_outliers-num_ignore_outliers):]
        inlier_idx_2 = sorted_outputs_idx_2[:(num_inliers-num_ignore_inliers)]
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

        print(f"F1 Score of contrastive learning model 1: {f1_score(gt_labels[model1_corr_labels_idx].cpu(), model1_preds[model1_corr_labels_idx].cpu())}")
        print(f"F1 Score of contrastive learning model 2: {f1_score(gt_labels[model2_corr_labels_idx].cpu(), model1_preds[model2_corr_labels_idx].cpu())}")

        model1_select_idx = torch.cat((model1_selected_clean_idx, model1_corr_labels_idx))
        model2_select_idx = torch.cat((model2_selected_clean_idx ,model2_corr_labels_idx))
        model1_select_idx = torch.unique(model1_select_idx)
        model2_select_idx = torch.unique(model2_select_idx)

        return model1_select_idx, model2_select_idx, model1_preds, model2_preds, model1_corr_labels_idx, gt_labels[model1_selected_clean_idx]
    



def run_unity(args, dataloader, dim, y_noisy, outlier_ratio, outlier_score, noise_rate_outlier, noise_rate_inlier):
    
    '''
    Starts running Unity
    
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
    print(f"Running Unity with experiemnt: {args.experiment}")
    trainer = Trainer(args, dim, outlier_ratio)  # initialize models
    trainer.train_unity(dataloader, (torch.eye(2, device='cuda:0')[y_noisy]), outlier_score, noise_rate_outlier, noise_rate_inlier)  # train models