import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from models.base_model_moons import BaseNet
from models.variational_inference_flipout_moons import BayesianNNFlipout

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=0.1):
        super(BayesianLinear, self).__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_logsigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-2.5))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_logsigma = nn.Parameter(torch.Tensor(out_features).fill_(-2.5))
        
        self.prior = Normal(0, prior_std)

    def forward(self, x):
        weight_epsilon = torch.randn_like(self.weight_logsigma)
        bias_epsilon = torch.randn_like(self.bias_logsigma)
        
        weight = self.weight_mu + torch.exp(self.weight_logsigma) * weight_epsilon
        bias = self.bias_mu + torch.exp(self.bias_logsigma) * bias_epsilon
        
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        weight_posterior = Normal(self.weight_mu, torch.exp(self.weight_logsigma))
        bias_posterior = Normal(self.bias_mu, torch.exp(self.bias_logsigma))
        
        weight_kl = kl_divergence(weight_posterior, self.prior).sum()
        bias_kl = kl_divergence(bias_posterior, self.prior).sum()
        
        return weight_kl + bias_kl

class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianNN, self).__init__()
        self.blinear1 = BayesianLinear(input_dim, hidden_dim)
        self.blinear2 = BayesianLinear(hidden_dim, hidden_dim)
        self.blinear3 = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.blinear1(x))
        x = torch.relu(self.blinear2(x))
        return self.blinear3(x)

    def kl_divergence(self):
        return self.blinear1.kl_divergence() + self.blinear2.kl_divergence() + self.blinear3.kl_divergence()

class VariationalInferenceNet:
    def __init__(self, input_dim, output_dim, hidden_dim=128, lr=1e-3):
        self.model = BayesianNNFlipout(input_dim, hidden_dim, output_dim)
        self.device = self.model.device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def fit(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        
        output = self.model(x)
        nll_loss = F.cross_entropy(output, y)
        kl_loss = self.model.kl_divergence()
        
        # Annealing factor for KL term
        beta = min(1.0, self.steps / 1000) if hasattr(self, 'steps') else 1.0
        self.steps = getattr(self, 'steps', 0) + 1
        
        loss = nll_loss + beta * kl_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def evaluate(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        with torch.no_grad():
            output = self.model(x)
            prediction_loss = F.cross_entropy(output, y)
            kl_loss = self.model.kl_divergence()
            loss = prediction_loss + self.kl_weight * kl_loss
            pred = output.argmax(dim=1)
            correct = pred.eq(y).sum().item()
            accuracy = correct / len(y)
        return loss.item(), accuracy

    def predict(self, x, n_samples=50):
        self.model.train() 
        predictions = []
        x = x.to(self.device)
        
        for _ in range(n_samples):
            with torch.no_grad():
                output = self.model(x)
                predictions.append(F.softmax(output, dim=1))
                
        predictions = torch.stack(predictions)
        mean = predictions.mean(0)
        variance = predictions.var(0)
        return mean, variance


    def sample_eval(self, x, y, n_samples=50):
        mean_pred, _ = self.predict(x, n_samples)
        y = y.to(self.device)
        loss = F.cross_entropy(mean_pred, y)
        pred = mean_pred.argmax(dim=1)
        correct = pred.eq(y).sum().item()
        accuracy = correct / len(y)
        return loss.item(), accuracy

    def save(self, filename):
        checkpoint = {
            'models_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'kl_weight': self.kl_weight,
        }
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['models_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.kl_weight = checkpoint['kl_weight']
        print(f"Model loaded from {filename}")


