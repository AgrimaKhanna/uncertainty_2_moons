from sklearn.datasets import make_moons
import torch
from torch.utils.data import TensorDataset

def create_moons_dataset(n_samples=200, noise=0.2):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X, y)

def get_data_loader(batch_size=32, n_samples=200):
    dataset = create_moons_dataset(n_samples)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
