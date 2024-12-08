from data.make_dataset_moons import get_data_loader
from models.mc_dropout_moons import MCDropout
from utils.training_utils_moons import train_model

dataloader = get_data_loader()
model = MCDropout()
train_model(model, dataloader)
