from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import math

__all__ = ["model"]


class model(nn.Module):
    def __init__(self, feature_dim=2, hidden_dim=5, num_classes=2, args=None):
        super(model, self).__init__()
        self.args = args

        if args.batch_norm == "true":
            self.fc1 = nn.Linear(feature_dim, 50)
            self.bn1 = nn.BatchNorm1d(50)
            self.fc2 = nn.Linear(50, 100)
            self.bn2 = nn.BatchNorm1d(100)
            self.fc3 = nn.Linear(100, 50)
            self.bn3 = nn.BatchNorm1d(50) 
            self.fc4 = nn.Linear(50, num_classes)
        else:
            self.fc1 = nn.Linear(feature_dim, 50)
            self.fc2 = nn.Linear(50, 100)
            self.fc3 = nn.Linear(100, 50)
            self.fc4 = nn.Linear(50, num_classes)

    def forward(self, x):
        if self.args.batch_norm == "true":
            out = F.relu(self.bn1(self.fc1(x)))
            out = F.relu(self.bn2(self.fc2(out)))
            out = F.relu(self.bn3(self.fc3(out)))
            out = self.fc4(out)
            return out
        else:
            out = F.relu(self.fc1(x))
            out = F.relu(self.fc2(out))
            out = F.relu(self.fc3(out))
            out = self.fc4(out)
            return out

class TwinNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super(TwinNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 8),  # nn.Linear(feature_dim, 8)
            nn.ReLU(),
            nn.Linear(8, 16),  # nn.Linear(8, 16)
            nn.ReLU(),
            nn.Linear(16, 8),  # nn.Linear(16, 8)
            nn.ReLU(),
            nn.Linear(8, 128)  # nn.Linear(8, 128)
        )

    def forward(self, x1, x2):
        output1 = self.fc(x1)
        output2 = self.fc(x2)
        return output1, output2

    def inference(self, x):
        return self.fc(x)