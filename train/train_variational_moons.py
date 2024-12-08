import torch
from data.make_dataset_moons import get_data_loader
from models.variational_inference_moons import BayesianNN
from utils.training_utils_moons import train_variational

# Set up data and model
dataloader = get_data_loader()
model = BayesianNN()
train_variational(model, dataloader)
