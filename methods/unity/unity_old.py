from torch import nn
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import math
from torch import autograd
import torch.nn.functional as F
from sklearn.preprocessing import PowerTransformer
import wandb
import csv
import os
from torch.optim.lr_scheduler import StepLR 
from torch import linalg

from methods.models.model import model
from methods.unity.utils import EMARecorder

class Trainer():
    def __init__(self, args, dim):
        self.args, self.dim = args, dim
        self.init_models()

        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_raw = nn.CrossEntropyLoss(reduction="none")


    def init_models(self):
        '''
        initialize models
        '''
        self.model1 = model(feature_dim=self.dim, hidden_dim=30, num_classes=2).cuda()
        self.model2 = model(feature_dim=self.dim, hidden_dim=30, num_classes=2).cuda()

        self.optimizer1 = torch.optim.SGD(self.model1.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.optimizer2 = torch.optim.SGD(self.model2.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        #self.scheduler1 = StepLR(self.optimizer1, step_size=5, gamma=0.3)
        #self.scheduler2 = StepLR(self.optimizer2, step_size=5, gamma=0.3)


    def inference_model(self, model, dataloader):
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

            confidences.append(pred_prob[:, 1])
            pred_labels.append(pred_label)
            true_labels.append(label)
            indices.append(idx)

        confidences = torch.cat(confidences, dim=0).detach().cpu().numpy()
        pred_labels = torch.cat(pred_labels, dim=0).detach().cpu().numpy()
        true_labels = torch.cat(true_labels, dim=0).detach().cpu().numpy()
        indices = torch.cat(indices, dim=0).detach().cpu().numpy()

        return confidences[np.argsort(indices)], pred_labels[np.argsort(indices)], true_labels[np.argsort(indices)]
    
    
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
            loss_model1 = (self.args.beta * self.model1_ema.get_loss_ema(idx)) + ((1-self.args.beta) * loss_model1)
            loss_model2 = (self.args.beta * self.model2_ema.get_loss_ema(idx)) + ((1-self.args.beta) * loss_model2)

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
    

    def get_adaptive_threshold_agreement(self, sampling_score):
        '''
        Calculates the adaptive threshold for agreement module

        Parameters
        ----------
        sampling_score: sampling scores for agreement module

        Returns
        -------
        threshold: adaptive threshold
        '''
        if len(sampling_score) > 1:
            threshold = max(torch.mean(sampling_score) - (torch.std(sampling_score)), min(sampling_score))
        else:
            threshold = sampling_score
        return threshold
    

    def get_adaptive_threshold_disagreement(self, sampling_score):
        '''
        Calculates the adaptive threshold for disagreement module

        Parameters
        ----------
        sampling_score: sampling scores for disagreement module

        Returns
        -------
        threshold: adaptive threshold        
        '''
        if len(sampling_score) > 1:
            #threshold = min(max(sampling_score), torch.mean(sampling_score) + (torch.std(sampling_score)))
            threshold = max(torch.mean(sampling_score) - (torch.std(sampling_score)), min(sampling_score))
        else:
            threshold = sampling_score
        return threshold


    def sample_selection(self, model1_pred_logits, model2_pred_logits, idx, epoch, y_noisy, model1_pred_label, model2_pred_label):
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
        # warmup by selecting all samples
        if epoch < self.args.epochs_warmup:
            model1_select_idx = torch.arange(0, len(idx)).cuda()  # select all data
            model2_select_idx = torch.arange(0, len(idx)).cuda()  

            # loss
            loss_model1 = self.loss_fn_raw(model1_pred_logits, y_noisy[idx])
            loss_model2 = self.loss_fn_raw(model2_pred_logits, y_noisy[idx])

            # ema of loss
            loss_model1, loss_model2 = self.ema_loss(epoch, loss_model1, loss_model2, idx)

            model1_agree_idx = model1_select_idx
            model2_agree_idx = model2_select_idx
            model1_disagree_idx = model1_select_idx
            model2_disagree_idx = model2_select_idx
        
        # Do sample selection
        else:
            alpha = self.args.correction_threshold_inlier  # tradeoff between loss and similarity score or confident score .5
            # loss
            loss_model1 = self.loss_fn_raw(model1_pred_logits, y_noisy[idx])
            loss_model2 = self.loss_fn_raw(model2_pred_logits, y_noisy[idx])
            
            # ema of loss            
            loss_model1, loss_model2 = self.ema_loss(epoch, loss_model1, loss_model2, idx)

            ###### Agreement module ######
            # JS divergence: smaller value has more similar distributions
            m = (model1_pred_logits.softmax(1) + model2_pred_logits.softmax(1))/2
            kl_main = F.kl_div(model1_pred_logits.log_softmax(1), m, reduction='none')
            kl_weight = F.kl_div(model2_pred_logits.log_softmax(1), m, reduction='none')
            similarity_score = 0.5 * torch.sum(kl_main, 1) + 0.5 * torch.sum(kl_weight, 1)

            # EMA of similarity score
            #similarity_score = self.ema_sim_score(epoch, similarity_score, idx)  # use EMA for similarity score
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
            sampling_score = torch.clamp(sampling_score, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            sampling_score = torch.from_numpy(power_transformer.fit_transform(sampling_score.detach().cpu().reshape(-1,1))).flatten().cuda()

            # adaptive threshold for how many samples to select as clean
            sample_score_agreement_inliers_threshold = self.get_adaptive_threshold_agreement(sampling_score[combined_inlier_idx])
            sample_score_agreement_outliers_threshold = self.get_adaptive_threshold_agreement(sampling_score[combined_outlier_idx])
        
            # select samples that are smaller than the threshold
            similar_smallest_losses_inliers_idx = torch.nonzero(sampling_score[combined_inlier_idx] <= sample_score_agreement_inliers_threshold).flatten()
            similar_smallest_losses_outliers_idx = torch.nonzero(sampling_score[combined_outlier_idx] <= sample_score_agreement_outliers_threshold).flatten()

            # selected indexes 
            model1_select_idx = torch.concat((combined_inlier_idx[similar_smallest_losses_inliers_idx], combined_outlier_idx[similar_smallest_losses_outliers_idx]))
            model2_select_idx = torch.concat((combined_inlier_idx[similar_smallest_losses_inliers_idx], combined_outlier_idx[similar_smallest_losses_outliers_idx]))

            model1_agree_idx = model1_select_idx
            model2_agree_idx = model2_select_idx        
            
            ###### Disagreement module ######
            # prob of sample being an outlier
            prob_model1 = model1_pred_logits.softmax(1)[:,1]
            prob_model2 = model2_pred_logits.softmax(1)[:,1]

            # EMA of probs
            #prob_model1, prob_model2 = self.ema_prob(epoch, prob_model1, prob_model2, idx)

            # calculate sampling scores. Large difference in confidence and small loss have large sampling scores
            confscore_model1 = (torch.abs(prob_model1 - .5) - torch.abs(prob_model2 - .5))
            confscore_model2 = (torch.abs(prob_model2 - .5) - torch.abs(prob_model1 - .5))

            # normalize the conf scores 
            confscore_model1 = (confscore_model1 - torch.min(confscore_model1)) / (torch.max(confscore_model1) - torch.min(confscore_model1))
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

            # Transform the sampling score distribution to gaussian distribution
            large_conf_small_loss_score_model1 = torch.clamp(large_conf_small_loss_score_model1, .000000001, 10)
            large_conf_small_loss_score_model2 = torch.clamp(large_conf_small_loss_score_model2, .000000001, 10)
            power_transformer = PowerTransformer(method='box-cox')
            large_conf_small_loss_score_model1 = torch.from_numpy(power_transformer.fit_transform(large_conf_small_loss_score_model1.detach().cpu().reshape(-1,1))).flatten().cuda()
            large_conf_small_loss_score_model2 = torch.from_numpy(power_transformer.fit_transform(large_conf_small_loss_score_model2.detach().cpu().reshape(-1,1))).flatten().cuda()
            
            large_conf_small_loss_score_model1_inlier = large_conf_small_loss_score_model1[largest_similarity_score_idx_model1_inlier]
            large_conf_small_loss_score_model1_outlier = large_conf_small_loss_score_model1[largest_similarity_score_idx_model1_outlier]
            large_conf_small_loss_score_model2_inlier = large_conf_small_loss_score_model2[largest_similarity_score_idx_model2_inlier]
            large_conf_small_loss_score_model2_outlier = large_conf_small_loss_score_model2[largest_similarity_score_idx_model2_outlier]

            # adaptive threshold
            large_conf_sample_score_model1_inlier_threshold = self.get_adaptive_threshold_disagreement(large_conf_small_loss_score_model1_inlier)
            large_conf_sample_score_model1_outlier_threshold = self.get_adaptive_threshold_disagreement(large_conf_small_loss_score_model1_outlier)
            large_conf_sample_score_model2_inlier_threshold = self.get_adaptive_threshold_disagreement(large_conf_small_loss_score_model2_inlier)
            large_conf_sample_score_model2_outlier_threshold = self.get_adaptive_threshold_disagreement(large_conf_small_loss_score_model2_outlier)

            # select samples that are smaller than the threshold
            large_conf_sample_score_model1_inlier_idx = torch.nonzero(large_conf_small_loss_score_model1_inlier <= large_conf_sample_score_model1_inlier_threshold).flatten()
            large_conf_sample_score_model1_outlier_idx = torch.nonzero(large_conf_small_loss_score_model1_outlier <= large_conf_sample_score_model1_outlier_threshold).flatten()
            large_conf_sample_score_model2_inlier_idx = torch.nonzero(large_conf_small_loss_score_model2_inlier <= large_conf_sample_score_model2_inlier_threshold).flatten()
            large_conf_sample_score_model2_outlier_idx = torch.nonzero(large_conf_small_loss_score_model2_outlier <= large_conf_sample_score_model2_outlier_threshold).flatten()

            model1_select_idx = torch.cat((model1_select_idx, largest_similarity_score_idx_model1_inlier[large_conf_sample_score_model1_inlier_idx], largest_similarity_score_idx_model1_outlier[large_conf_sample_score_model1_outlier_idx]))
            model2_select_idx = torch.cat((model2_select_idx, largest_similarity_score_idx_model2_inlier[large_conf_sample_score_model2_inlier_idx], largest_similarity_score_idx_model2_outlier[large_conf_sample_score_model2_outlier_idx]))

            model1_disagree_idx = torch.cat([largest_similarity_score_idx_model1_inlier[large_conf_sample_score_model1_inlier_idx], largest_similarity_score_idx_model1_outlier[large_conf_sample_score_model1_outlier_idx]])
            model2_disagree_idx = torch.cat([largest_similarity_score_idx_model2_inlier[large_conf_sample_score_model2_inlier_idx], largest_similarity_score_idx_model2_outlier[large_conf_sample_score_model2_outlier_idx]])
        
        return model1_select_idx, model2_select_idx, model1_agree_idx, model2_agree_idx, model1_disagree_idx, model2_disagree_idx
    

    def correct_labels(self, model1_select_idx, model2_select_idx, model1_pred_label, model2_pred_label, second_order_gradients_1, second_order_gradients_2, idx, y_noisy, model1_pred_logits, model2_pred_logits):
        '''
        Uses 2nd gradients w.r.t X to find additional samples 
        Samples with small 2nd gradients are selected as inliers
        Samples with large 2nd gradients are selected as outliers

        Parameters
        ----------
        model1_select_idx: indexes of samples selected by model 1
        model2_select_idx: indexes of samples selected by model 2
        model1_pred_label: model 1's predicted labels
        model2_pred_label: model 2's predicted labels
        second_order_gradients_1: model 1's second order gradients
        second_order_gradients_2: model 2's second order gradients
        idx: indexes of samples in the mini batch
        y_noisy : initial noisy pseudo labels

        Returns
        -------
        model1_select_idx: updated indexes of samples selected by model 1
        model2_select_idx: updated indexes of samples selected by model 2
        y_noisy: Corrected noisy pseudo labels
        model1_pred_label: Corrected model 1's predicted labels
        model2_pred_label: Corrected model 2's predicted labels
        '''
        # # loss 
        # loss_model1 = self.loss_fn_raw(model1_pred_logits, y_noisy[idx])
        # loss_model2 = self.loss_fn_raw(model2_pred_logits, y_noisy[idx])
        # # normalize
        # loss_model1 = (loss_model1 - torch.min(loss_model1)) / (torch.max(loss_model1) - torch.min(loss_model1))
        # loss_model2 = (loss_model2 - torch.min(loss_model2)) / (torch.max(loss_model2) - torch.min(loss_model2))   
        
        # L2 norm of second order gradients
        model1_2grad_l2 = linalg.vector_norm(second_order_gradients_1, ord=2, dim=1) 
        model2_2grad_l2 = linalg.vector_norm(second_order_gradients_2, ord=2, dim=1)
        
        # # normalize
        # model1_2grad_l2 = (model1_2grad_l2 - torch.min(model1_2grad_l2)) / (torch.max(model1_2grad_l2) - torch.min(model1_2grad_l2))
        # model2_2grad_l2 = (model2_2grad_l2 - torch.min(model2_2grad_l2)) / (torch.max(model2_2grad_l2) - torch.min(model2_2grad_l2))

        # # Transform the grad distribution to gaussian distribution
        # model1_2grad_l2 = torch.clamp(model1_2grad_l2, .000000001, 10)
        # power_transformer = PowerTransformer(method='box-cox')
        # model1_2grad_l2 = torch.from_numpy(power_transformer.fit_transform(model1_2grad_l2.detach().cpu().reshape(-1,1))).flatten().cuda()

        # model2_2grad_l2 = torch.clamp(model2_2grad_l2, .000000001, 10)
        # model2_2grad_l2 = torch.from_numpy(power_transformer.fit_transform(model2_2grad_l2.detach().cpu().reshape(-1,1))).flatten().cuda()
        
        index_not_selected_model1 = torch.nonzero(~torch.isin(torch.arange(len(idx)).cuda(), model1_select_idx)).flatten()
        index_not_selected_model2 = torch.nonzero(~torch.isin(torch.arange(len(idx)).cuda(), model2_select_idx)).flatten()

        # get eligible samples
        model1_2grad_l2 = model1_2grad_l2[index_not_selected_model1]
        model2_2grad_l2 = model2_2grad_l2[index_not_selected_model2]

        # calculate threshold 
        model1_threshold_inlier = torch.mean(model1_2grad_l2) + (self.args.correction_threshold_outlier * torch.std(model1_2grad_l2))
        model1_threshold_outlier = torch.mean(model1_2grad_l2) - (self.args.correction_threshold_outlier * torch.std(model1_2grad_l2))
        model2_threshold_inlier = torch.mean(model2_2grad_l2) + (self.args.correction_threshold_outlier * torch.std(model2_2grad_l2))
        model2_threshold_outlier = torch.mean(model2_2grad_l2) - (self.args.correction_threshold_outlier * torch.std(model2_2grad_l2))

        # select at least one sample
        if len(model1_2grad_l2) != 0:
            model1_threshold_inlier = min(model1_threshold_inlier, torch.max(model1_2grad_l2))
            model1_threshold_outlier = max(model1_threshold_outlier, torch.min(model1_2grad_l2))
        if len(model2_2grad_l2) != 0:
            model2_threshold_inlier = min(model2_threshold_inlier, torch.max(model2_2grad_l2))
            model2_threshold_outlier = max(model2_threshold_outlier, torch.min(model2_2grad_l2))
        
        # get index of inlier refurbished samples 
        model1_inlier_idx = torch.nonzero(model1_2grad_l2 >= model1_threshold_inlier).flatten()
        model2_inlier_idx = torch.nonzero(model2_2grad_l2 >= model2_threshold_inlier).flatten()

        # get index of outlier refurbished samples
        model1_outlier_idx = torch.nonzero(model1_2grad_l2 <= model1_threshold_outlier).flatten()
        model2_outlier_idx = torch.nonzero(model2_2grad_l2 <= model2_threshold_outlier).flatten()    
        
        selected_inliers_model1_idx = index_not_selected_model1[model1_inlier_idx]
        selected_inliers_model2_idx = index_not_selected_model2[model2_inlier_idx]
        selected_outliers_model1_idx = index_not_selected_model1[model1_outlier_idx]
        selected_outliers_model2_idx = index_not_selected_model2[model2_outlier_idx]
        
        
        # # how many samples should be selected: none threshing
        # num_samples_correct_model1_inlier = math.ceil(self.args.correction_threshold_inlier * len(ranked_indices_model1))
        # num_samples_correct_model2_inlier = math.ceil(self.args.correction_threshold_inlier * len(ranked_indices_model2))
        # num_samples_correct_model1_outlier = math.ceil(self.args.correction_threshold_outlier * len(ranked_indices_model1))
        # num_samples_correct_model2_outlier = math.ceil(self.args.correction_threshold_outlier * len(ranked_indices_model2))

        # # select the k samples with the largest grads as inliers and the k samples with the smallest grads as outliers
        # if num_samples_correct_model1_inlier == 0:
        #     selected_inliers_model1_idx = torch.tensor([], device=model1_select_idx.device, dtype=int)
        # else:
        #     selected_inliers_model1_idx = index_not_selected_model1[ranked_indices_model1[-num_samples_correct_model1_inlier:]]
        
        # if num_samples_correct_model2_inlier == 0:
        #     selected_inliers_model2_idx = torch.tensor([], device=model1_select_idx.device, dtype=int)
        # else:
        #     selected_inliers_model2_idx = index_not_selected_model2[ranked_indices_model2[-num_samples_correct_model2_inlier:]]
        
        # if num_samples_correct_model1_outlier == 0:
        #     selected_outliers_model1_idx = torch.tensor([], device=model1_select_idx.device, dtype=int)
        # else:
        #     selected_outliers_model1_idx = index_not_selected_model1[ranked_indices_model1[:num_samples_correct_model1_outlier]]
        
        # if num_samples_correct_model2_outlier == 0:
        #     selected_outliers_model2_idx = torch.tensor([], device=model1_select_idx.device, dtype=int)
        # else:
        #     selected_outliers_model2_idx = index_not_selected_model2[ranked_indices_model2[:num_samples_correct_model2_outlier]]

        # correct labels and combine with previous selected samples
        model1_select_idx = torch.cat((model1_select_idx, selected_inliers_model1_idx, selected_outliers_model1_idx))
        model2_select_idx = torch.cat((model2_select_idx, selected_inliers_model2_idx, selected_outliers_model2_idx))

        model1_select_idx = torch.unique(model1_select_idx)
        model2_select_idx = torch.unique(model2_select_idx)

        # update the noisy labels
        temp_mv_labels = y_noisy[idx]
        temp_mv_labels[selected_inliers_model1_idx.cpu()] = torch.eye(2)[np.zeros(len(selected_inliers_model1_idx), dtype=int)].cuda()
        temp_mv_labels[selected_inliers_model2_idx.cpu()] = torch.eye(2)[np.zeros(len(selected_inliers_model2_idx), dtype=int)].cuda()
        temp_mv_labels[selected_outliers_model1_idx.cpu()] = torch.eye(2)[np.ones(len(selected_outliers_model1_idx), dtype=int)].cuda()
        temp_mv_labels[selected_outliers_model2_idx.cpu()] = torch.eye(2)[np.ones(len(selected_outliers_model2_idx), dtype=int)].cuda()
        y_noisy[idx] = temp_mv_labels

        model1_pred_label_temp = np.array(model1_pred_label.cpu())
        model2_pred_label_temp = np.array(model2_pred_label.cpu())
        
        model1_pred_label_temp[selected_inliers_model1_idx.cpu()] = np.zeros(len(selected_inliers_model1_idx), dtype=int)
        model2_pred_label_temp[selected_inliers_model2_idx.cpu()] = np.zeros(len(selected_inliers_model2_idx), dtype=int)
        model1_pred_label_temp[selected_outliers_model1_idx.cpu()] = np.ones(len(selected_outliers_model1_idx), dtype=int)
        model2_pred_label_temp[selected_outliers_model2_idx.cpu()] = np.ones(len(selected_outliers_model2_idx), dtype=int)

        model1_pred_label = torch.tensor(model1_pred_label_temp).cuda()
        model2_pred_label = torch.tensor(model2_pred_label_temp).cuda()

        model1_corrected_idx = torch.cat((selected_inliers_model1_idx, selected_outliers_model1_idx))
        model2_corrected_idx = torch.cat((selected_inliers_model2_idx, selected_outliers_model2_idx))

        return model1_select_idx, model2_select_idx, y_noisy, model1_pred_label, model2_pred_label, model1_corrected_idx, model2_corrected_idx
    
    
    def unity_sgd(self, dataloader, y_noisy, epoch):
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
        model1_correction_pred = []
        model1_correction_gt = []
        model2_correction_pred = []
        model2_correction_gt = []

        # loop through batches
        for iter_num, (idx, data, label) in enumerate(dataloader):
            data, label = data.cuda(), label.cuda()

            x_1 = data.requires_grad_(True) # track gradients w.r.t. x
            x_2 = data.requires_grad_(True)  # track gradients w.r.t. x

            # pass data through model1 for gradient calculation
            model1_pred_logits = self.model1(x_1)
            model1_pred_prob = torch.softmax(model1_pred_logits, dim=1)
            model1_pred_label = torch.argmax(model1_pred_prob, dim=1)

            # pass data through model2 for gradient calculation
            model2_pred_logits = self.model2(x_2)
            model2_pred_prob = torch.softmax(model2_pred_logits, dim=1)
            model2_pred_label = torch.argmax(model2_pred_prob, dim=1)

            model1_logits_outliers = model1_pred_logits[:,1]
            model2_logits_outliers = model2_pred_logits[:,1]

            grad_outputs_1_model1 = torch.ones_like(model1_logits_outliers, device=data.device, requires_grad=False)
            grad_outputs_1_model2 = torch.ones_like(model2_logits_outliers, device=data.device, requires_grad=False)

            # Compute first order Gradients
            gradients_data_1 = autograd.grad(
                outputs=model1_logits_outliers,
                inputs=x_1,
                grad_outputs=grad_outputs_1_model1,
                create_graph=True,
                retain_graph=True,
            )[0]
            gradients_data_2 = autograd.grad(
                outputs=model2_logits_outliers,
                inputs=x_2,
                grad_outputs=grad_outputs_1_model2,
                create_graph=True,
                retain_graph=True,
            )[0]

            grad_outputs_2_model1 = torch.ones_like(gradients_data_1, device=data.device, requires_grad=False)
            grad_outputs_2_model2 = torch.ones_like(gradients_data_2, device=data.device, requires_grad=False)

            # compute second order gradients
            second_order_gradients_1 = autograd.grad(
                outputs = gradients_data_1, 
                inputs = x_1,
                grad_outputs = grad_outputs_2_model1,
                create_graph = True,
                retain_graph=False
            )[0]
            second_order_gradients_2 = autograd.grad(
                outputs = gradients_data_2, 
                inputs = x_2,
                grad_outputs = grad_outputs_2_model2,
                create_graph = True,
                retain_graph=False
            )[0]

            # clear gradients
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            
            # pass data through model1
            model1_pred_logits = self.model1(data)
            model1_pred_prob = torch.softmax(model1_pred_logits, dim=1)
            model1_pred_label = torch.argmax(model1_pred_prob, dim=1)

            # pass data through model2
            model2_pred_logits = self.model2(data)
            model2_pred_prob = torch.softmax(model2_pred_logits, dim=1)
            model2_pred_label = torch.argmax(model2_pred_prob, dim=1)

            
            # Sample selection
            model1_select_idx, model2_select_idx, model1_agree_idx, model2_agree_idx, model1_disagree_idx, model2_disagree_idx = self.sample_selection(model1_pred_logits, model2_pred_logits, idx, epoch, y_noisy, model1_pred_label, model2_pred_label)

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
            if not epoch < self.args.epochs_warmup:
                model1_select_idx, model2_select_idx, y_noisy, model1_pred_label, model2_pred_label, model1_corrected_idx, model2_corrected_idx = self.correct_labels(model1_select_idx, model2_select_idx, model1_pred_label, model2_pred_label, second_order_gradients_1, second_order_gradients_2, idx, y_noisy, model1_pred_logits, model2_pred_logits)

                # track performance of correction
                model1_correction_pred.append(model1_pred_label[model1_corrected_idx])
                model2_correction_pred.append(model2_pred_label[model2_corrected_idx])
                model1_correction_gt.append(label[model1_corrected_idx])
                model2_correction_gt.append(label[model2_corrected_idx])

            else:
                model1_correction_pred.append(model1_pred_label[model1_select_idx])
                model2_correction_pred.append(model2_pred_label[model2_select_idx])
                model1_correction_gt.append(label[model1_select_idx])
                model2_correction_gt.append(label[model2_select_idx])

            # Model updating 
            if epoch < self.args.epochs_warmup:  # use noisy labels for warmup
                model2_cf_data, model2_cf_label = data[model2_select_idx], torch.argmax(y_noisy[idx], axis=1)
                model1_cf_data, model1_cf_label = data[model1_select_idx], torch.argmax(y_noisy[idx], axis=1)
            else:  # use model predictions
                model1_cf_data, model1_cf_label = data[model1_select_idx], model1_pred_label[model1_select_idx]
                model2_cf_data, model2_cf_label = data[model2_select_idx], model2_pred_label[model2_select_idx] 

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

        return y_noisy, model1_agreement_pred, model2_agreement_pred, model1_agreement_gt, model2_agreement_gt, model1_disagreement_pred, model2_disagreement_pred, model1_disagreement_gt, model2_disagreement_gt, model1_correction_pred, model2_correction_pred, model1_correction_gt, model2_correction_gt


    def train_unity(self, dataloader, y_noisy):
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
        # Store best metrics
        max_model1_f1 = 0
        max_model2_f1 = 0
        best_epoch_model1 = 0
        best_epoch_model2 = 0
        max_model1_ROCAUC = 0 
        max_model2_ROCAUC = 0

        # Store history of losses and confidences
        self.model1_ema = EMARecorder(y_noisy)
        self.model2_ema = EMARecorder(y_noisy)

        for epoch in range(self.args.epochs):
            print(f"####### Epoch {epoch} #######")
            self.model1.train()
            self.model2.train()

            # SGD
            y_noisy_updated, model1_agreement_pred, model2_agreement_pred, model1_agreement_gt, model2_agreement_gt, model1_disagreement_pred, model2_disagreement_pred, model1_disagreement_gt, model2_disagreement_gt, model1_correction_pred, model2_correction_pred, model1_correction_gt, model2_correction_gt = self.unity_sgd(dataloader, y_noisy, epoch)
            y_noisy = y_noisy_updated

            # inference model 1
            model1_confidences, model1_pred_labels, true_labels = self.inference_model(1, dataloader)

            # inference model 2
            model2_confidences, model2_pred_labels, true_labels = self.inference_model(2, dataloader)

            # report performance
            model1_f1 = f1_score(true_labels, model1_pred_labels)
            if model1_f1 > max_model1_f1:
                best_epoch_model1 = epoch
            print(f"F1 score of model1 is {model1_f1}")
            max_model1_f1 = max(max_model1_f1, model1_f1)
            model1_auc = roc_auc_score(true_labels, model1_confidences)
            print(f"ROC AUC of model1 is {model1_auc}")
            max_model1_ROCAUC = max(max_model1_ROCAUC, model1_auc)

            model2_f1 = f1_score(true_labels, model2_pred_labels)
            if model2_f1 > max_model2_f1:
                best_epoch_model2 = epoch                
            print(f"F1 score of model2 is {model2_f1}")
            max_model2_f1 = max(max_model2_f1, model2_f1)
            model2_auc = roc_auc_score(true_labels, model2_confidences)
            print(f"ROC AUC of model2 is {model2_auc}")
            max_model2_ROCAUC = max(max_model2_ROCAUC, model2_auc)

            model1_agreementf1 = f1_score(torch.cat(model1_agreement_gt, dim=0).cpu(), torch.cat(model1_agreement_pred, dim=0).cpu())
            model2_agreementf1 = f1_score(torch.cat(model2_agreement_gt, dim=0).cpu(), torch.cat(model2_agreement_pred, dim=0).cpu())
            model1_disagreementf1 = f1_score(torch.cat(model1_disagreement_gt, dim=0).cpu(), torch.cat(model1_disagreement_pred, dim=0).cpu())
            model2_disagreementf1 = f1_score(torch.cat(model2_disagreement_gt, dim=0).cpu(), torch.cat(model2_disagreement_pred, dim=0).cpu())

            model1_correctionf1 = f1_score(torch.cat(model1_correction_gt, dim=0).cpu(), torch.cat(model1_correction_pred, dim=0).cpu())
            model2_correctionf1 = f1_score(torch.cat(model2_correction_gt, dim=0).cpu(), torch.cat(model2_correction_pred, dim=0).cpu())
                        
            if self.args.w_b == 1:
                wandb.log({'model1_f1': model1_f1,'model2_f1': model2_f1, 'model1_agreement_f1': model1_agreementf1, 'model2_agreement_f1': model2_agreementf1, 
                           'model1_disagreement_f1': model1_disagreementf1, 'model2_disagreement_f1': model2_disagreementf1, 'model1_correction_f1': model1_correctionf1, 
                           'model2_correction_f1': model2_correctionf1, 'count_agreement_model1' : len(torch.cat(model1_agreement_pred, dim=0)), 'count_agreement_model2' : len(torch.cat(model2_agreement_pred, dim=0)),
                           'count_disagreement_model1' : len(torch.cat(model1_disagreement_pred, dim=0)), 'count_disagreement_model2' : len(torch.cat(model2_disagreement_pred, dim=0)), 
                           'count_correction_model1' : len(torch.cat(model1_correction_pred, dim=0)), 'count_correction_model2' : len(torch.cat(model2_correction_pred, dim=0))})

            # adjust learning rate
            epoch_start_decay = 50
            if epoch_start_decay <= epoch:
                new_lr = (self.args.epochs - epoch) / (self.args.epochs - epoch_start_decay) * self.args.lr
                for param_group in self.optimizer1.param_groups:
                    param_group['lr'] = new_lr
                for param_group in self.optimizer2.param_groups:
                    param_group['lr'] = new_lr
            
            # self.scheduler1.step()
            # self.scheduler2.step()
        
                
        print("Done training Unity: Below are the max scores")
        print(f"Max F1 score of model1 is {max_model1_f1}")
        print(f"Max ROC AUC of model1 is {max_model1_ROCAUC}")
        print(f"Max F1 score of model2 is {max_model2_f1}")
        print(f"Max ROC AUC of model2 is {max_model2_ROCAUC}")

        # write results to csv
        csv_file_path = "results/results.csv"
        path_exists = False
        if os.path.exists(csv_file_path):
            path_exists = True
        
        with open(csv_file_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            if not path_exists:
                csv_writer.writerow(["Method", "Dataset", "seed", "LR", "Warmup", "beta", "correction_threshold_outlier", "correction_threshold_inlier",
                                     "Best Epoch model1", "Best F1 model1", "Best Epoch model2", "Best F1 model2", "Last F1 model1", "Last F1 model2"])
            csv_writer.writerow([self.args.method, self.args.dataset, self.args.seed, self.args.lr, self.args.epochs_warmup, self.args.beta,
                                 self.args.correction_threshold_outlier, self.args.correction_threshold_inlier, best_epoch_model1, max_model1_f1, 
                                 best_epoch_model2, max_model2_f1, model1_f1, model2_f1])

def run_unity(args, dataloader, dim, y_noisy):
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
    print("Running Unity")
    trainer = Trainer(args, dim)  # initialize models
    trainer.train_unity(dataloader, (torch.eye(2, device='cuda:0')[y_noisy]))  # train models