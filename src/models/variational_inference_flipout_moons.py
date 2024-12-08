import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np


class FlipoutLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1):
        super(FlipoutLinear, self).__init__()
        # Mean and log-variance parameters for variational weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.9))
        self.weight_logsigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-2.5))
        
        # Bias terms
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.9))
        self.bias_logsigma = nn.Parameter(torch.Tensor(out_features).fill_(-2.5))

        # Prior distribution 
        self.prior = Normal(0, prior_std)

    def forward(self, x):
        weight_epsilon = torch.randn_like(self.weight_logsigma)
        bias_epsilon = torch.randn_like(self.bias_logsigma)
        
        #Flipout trick: create perturbations for each weight sample
        weight = self.weight_mu + torch.exp(self.weight_logsigma) * weight_epsilon
        bias = self.bias_mu + torch.exp(self.bias_logsigma) * bias_epsilon
        
        return F.linear(x, weight, bias) 
    

    def kl_divergence(self):
        # Compute KL divergence between the posterior and prior distributions
        weight_posterior = Normal(self.weight_mu, torch.exp(self.weight_logsigma))
        bias_posterior = Normal(self.bias_mu, torch.exp(self.bias_logsigma))
        
        weight_kl = kl_divergence(weight_posterior, self.prior).sum()
        bias_kl = kl_divergence(bias_posterior, self.prior).sum()
        
        return weight_kl + bias_kl

class BayesianNNFlipout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianNNFlipout, self).__init__()
       
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.fc1 = FlipoutLinear(input_dim, hidden_dim)
        self.fc2 = FlipoutLinear(hidden_dim, hidden_dim)
        self.fc3 = FlipoutLinear(hidden_dim, output_dim)

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def kl_divergence(self):
        total_kl = sum(m.kl_divergence() for m in self.modules() if isinstance(m, FlipoutLinear))
        batch_size = 128  
        return total_kl / batch_size

    def predict_mean_and_variance(self, x, n_samples=50):
        x = x.to(self.device) 
        predictions = [self.forward(x).detach() for _ in range(n_samples)]
        mean = torch.stack(predictions).mean(0)
        variance = torch.stack(predictions).var(0)
        return mean, variance
    
