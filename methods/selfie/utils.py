import torch 

class ConfRecorder:
    '''
    Records the confidence of each sample
    '''
    def __init__(self, total_epochs, labels):
        self.labels = labels
        self.epoch = 0

        self.hist_preds = torch.zeros((total_epochs, len(self.labels))).cuda().detach()-1
    def record_conf(self, idx, preds):
        self.hist_preds[self.epoch, idx] = preds.clone().float().cuda().detach()
    def update_epoch(self, epoch):
        self.epoch = epoch
    def get_hist_conf(self, idx):
        return self.hist_preds[:self.epoch+1,idx]

