import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid=128):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, output_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)

class BaseNet:
    def __init__(self, model, lr=1e-3, weight_decay=0, opt_type="Adam"):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None
        self.epoch = 0
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Move model to the chosen device
        self.model.to(self.device)

        # Initialize optimizer by default
        self.create_opt(opt_type=opt_type)

    def create_opt(self, opt_type="SGD", momentum=0.5):
        if opt_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=momentum, weight_decay=self.weight_decay)
        elif opt_type == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def create_opt(self, opt_type="SGD", momentum=0.5):
        if opt_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=momentum, weight_decay=self.weight_decay)
        elif opt_type == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def update_lr(self, gamma=0.99):
        self.lr *= gamma
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def save(self, filename):
        torch.save({
            'epoch': self.epoch,
            'lr': self.lr,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.lr = checkpoint['lr']
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"Model loaded from {filename}")

    def fit(self, x, y):
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, x, y):
        x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            output = self.model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(y.view_as(pred)).sum().item()
            accuracy = correct / len(y)
        return loss.item(), accuracy

# Example usage
if __name__ == "__main__":
    input_dim = 100
    output_dim = 10
    model = CustomModel(input_dim=input_dim, output_dim=output_dim)
    base_net = BaseNet(model=model)
    print(f"Number of parameters: {base_net.get_nb_parameters()}")
