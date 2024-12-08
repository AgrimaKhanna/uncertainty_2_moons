from data.make_dataset_moons import get_data_loader
from models.deep_ensemble_moons import DeepEnsemble

dataloader = get_data_loader()
ensemble = DeepEnsemble(n_models=5)
ensemble.train_ensemble(dataloader)
