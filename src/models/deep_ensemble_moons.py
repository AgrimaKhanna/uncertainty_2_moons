import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid=128):
        super(EnsembleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, output_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)

class DeepEnsembleNet:
    def __init__(self, input_dim, output_dim, n_models=5, n_hid=128, lr=1e-3, weight_decay=0):
        self.models = [EnsembleModel(input_dim, output_dim, n_hid) for _ in range(n_models)]
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.lr = lr
        self.weight_decay = weight_decay

        # Initialize optimizers for each model
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            for model in self.models
        ]
        
        for model in self.models:
            model.to(self.device)

    def fit(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)

    def evaluate(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        total_loss, total_correct = 0.0, 0
        with torch.no_grad():
            for model in self.models:
                model.eval()
                output = model(x)
                loss = nn.CrossEntropyLoss()(output, y)
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                total_correct += pred.eq(y).sum().item()
        average_loss = total_loss / len(self.models)
        accuracy = total_correct / (len(y) * len(self.models))
        return average_loss, accuracy

    def predict(self, x):
        x = x.to(self.device)
        predictions = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                predictions.append(model(x))
        # Calculate mean and variance across the ensemble
        mean_prediction = torch.stack(predictions).mean(dim=0)
        variance = torch.stack(predictions).var(dim=0)
        return mean_prediction, variance

    def save(self, filename):
        checkpoint = {
            'models_state': [model.state_dict() for model in self.models],
            'optimizers_state': [opt.state_dict() for opt in self.optimizers],
            'lr': self.lr,
            'weight_decay': self.weight_decay
        }
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        for model, state_dict in zip(self.models, checkpoint['models_state']):
            model.load_state_dict(state_dict)
        for opt, state_dict in zip(self.optimizers, checkpoint['optimizers_state']):
            opt.load_state_dict(state_dict)
        print(f"Model loaded from {filename}")
