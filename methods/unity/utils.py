import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class EMARecorder:
    '''
    Records the EMA loss and confidences for each sample
    '''
    def __init__(self, y_noisy):
        self.hist_loss_ema = torch.zeros((1, len(y_noisy))).cpu().detach()-1
        self.hist_sim_ema = torch.zeros((1, len(y_noisy))).cuda().detach()-1
        self.hist_prob_ema = torch.zeros((1, len(y_noisy))).cuda().detach()-1

    def record_loss_ema(self, idx, loss):
        self.hist_loss_ema[0, idx] = loss.clone().float().cpu().detach()
    def record_sim_ema(self, idx, sim):
        self.hist_sim_ema[0, idx] = sim.clone().float().cuda().detach()

    def get_loss_ema(self, idx):
        return self.hist_loss_ema[0, idx]
    def get_sim_ema(self, idx):
        return self.hist_sim_ema[0, idx]
    
    def get_prob_ema(self, idx):
        return self.hist_prob_ema[0, idx]
    def record_prob_ema(self, idx, prob):
        self.hist_prob_ema[0, idx] = prob.clone().float().cuda().detach()


class PredsRecorder:
    '''
    Records the confidences for each sample at each training epoch
    '''
    def __init__(self, total_epochs, labels):

        self.labels = labels
        self.epoch = 0

        self.hist_preds = torch.zeros((total_epochs, len(self.labels)))-1

    def record_preds(self, idx, preds):
        self.hist_preds[self.epoch, idx] = preds.clone().float().cpu()

    def update_epoch(self, epoch):
        self.epoch = epoch

    def get_hist_preds(self, idx):
        return self.hist_preds[:self.epoch+1,idx]


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

# Custom Dataset
class CustomDatasetCont(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]