import matplotlib.pyplot as plt
from data.make_dataset_moons import create_moons_dataset
from models.mc_dropout_moons import MC_DropoutNet
from models.deep_ensemble_moons import DeepEnsembleNet
from models.variational_inference_moons import VariationalInferenceNet
from models.base_model_moons import BaseNet
from models.base_model_moons import Model
from utils.visualization_moons import plot_decision_boundary
import torch

# Sample sizes to evaluate
sample_sizes = [50, 100, 150, 200]

model_base = lambda input_dim, output_dim: Model(input_dim=input_dim, output_dim=output_dim)

models = {
    "Base Model": lambda input_dim, output_dim: BaseNet(model=Model(input_dim=input_dim, output_dim=output_dim)) ,
    "MC Dropout": lambda input_dim, output_dim: MC_DropoutNet(input_dim=input_dim, output_dim=output_dim, hidden_dim=128),
    "Deep Ensemble": lambda input_dim, output_dim: DeepEnsembleNet(input_dim=input_dim, output_dim=output_dim, n_models=5, n_hid=128),
    "Variational Inference": lambda input_dim, output_dim: VariationalInferenceNet(input_dim=input_dim, output_dim=output_dim, hidden_dim=128)
}

# Training parameters
epochs = 100
lr = 0.001


fig, axes = plt.subplots(len(models), len(sample_sizes), figsize=(15, 10))
fig.suptitle("Uncertainty Estimation Comparison with Varying Sample Sizes", fontsize=16)

input_dim = 2  
output_dim = 2

for row_idx, (model_name, ModelClass) in enumerate(models.items()):
    for col_idx, sample_size in enumerate(sample_sizes):

        dataset = create_moons_dataset(n_samples=sample_size * 2)  
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=sample_size, shuffle=True)

        model = ModelClass(input_dim, output_dim)

        # Train the model
        for epoch in range(epochs):
            for x_batch, y_batch in dataloader:
                if model_name == "Variational Inference":
                    loss = model.fit(x_batch, y_batch)
                else:
                    loss = model.fit(x_batch, y_batch)

        ax = axes[row_idx, col_idx]
        n_samples = 100 if model_name in ["MC Dropout", "Variational Inference"] else 1
        plot_decision_boundary(model, dataset[:][0], dataset[:][1], method=model_name.lower().replace(" ", "_"), ax=ax, n_samples=n_samples)
        ax.set_title(f"{sample_size} Samples")
        if col_idx == 0:
            ax.set_ylabel(model_name)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.savefig("uncertainty_comparison.png")
plt.show()
