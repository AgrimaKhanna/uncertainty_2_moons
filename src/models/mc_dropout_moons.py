import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model_moons import BaseNet

def MC_dropout(x, p=0.5, mask=True):
    return F.dropout(x, p=p, training=mask)

class MC_DropoutLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, pdrop=0.5):
        super(MC_DropoutLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = pdrop
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, sample=True):
        mask = self.training or sample  # Apply dropout if training or sampling
        x = MC_dropout(self.fc1(x), p=self.dropout, mask=mask)
        x = self.activation(x)
        x = MC_dropout(self.fc2(x), p=self.dropout, mask=mask)
        x = self.activation(x)
        return self.fc3(x)

class MC_DropoutNet(BaseNet):
    def __init__(self, input_dim, output_dim, hidden_dim=128, lr=1e-3, weight_decay=0):
        model = MC_DropoutLayer(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, pdrop=0.5)
        super().__init__(model=model, lr=lr, weight_decay=weight_decay)
        self.create_opt("Adam")

    def fit(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        loss = super().fit(x, y)
        return loss

    def evaluate(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        loss, accuracy = super().evaluate(x, y)
        return loss, accuracy

    def predict(self, x, n_samples=50):
        # Perform multiple forward passes to calculate mean and variance
        self.model.train()  # Keep dropout active during inference
        predictions = [self.model(x).detach() for _ in range(n_samples)]
        mean = torch.stack(predictions).mean(0)
        variance = torch.stack(predictions).var(0)
        return mean, variance
