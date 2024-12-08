import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_decision_boundary(model, X, y, method="mc_dropout", ax=None, n_samples=1):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(model.device)

    if hasattr(model, 'predict') and method in ["mc_dropout", "variational_inference"]:
        mean, var = model.predict(grid, n_samples=n_samples)
    elif hasattr(model, 'predict') and method == "deep_ensemble":
        mean, var = model.predict(grid)  
    else:
        if hasattr(model, 'model'):
            mean = model.model(grid).detach()  
        else:
            mean = model(grid).detach()
        var = torch.zeros_like(mean) 

    mean_class1 = torch.softmax(mean.detach(), dim=1)[:, 1].cpu().numpy().reshape(xx.shape)  

    if ax is None:
        ax = plt.gca()
    contour = ax.contourf(xx, yy, mean_class1, alpha=0.7, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=20)
    ax.set_xticks([])
    ax.set_yticks([])
    return contour
