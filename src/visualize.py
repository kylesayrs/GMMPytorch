import torch
import numpy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from torch.distributions.utils import logits_to_probs


COLORS = ["red", "blue", "green", "orange", "purple"]


def log_and_visualize(
    iteration_index: int,
    data: torch.Tensor,
    model: torch.nn.Module,
    loss: torch.Tensor,
    width: float
):
    print(f"Iteration: {iteration_index:2d}, Loss: {loss.item():.2f}")
    if data.shape[1] == 2:
        plot_2d(data, model, width)


def plot_2d(data: torch.Tensor, model: torch.nn.Module, width: float):    
    probs = logits_to_probs(model.mixture.logits).detach()
    covariance_matrices = model.get_covariance_matrix().detach()
    means = model.mus.detach()

    probs = torch.sqrt(probs)  # clearer visualization

    for cluster_index in range(covariance_matrices.shape[0]):
        x = numpy.linspace(-width - 1, width + 1, num=100)
        y = numpy.linspace(-width - 1, width + 1, num=100)
        X, Y = numpy.meshgrid(x, y)

        distr = multivariate_normal(
            cov=covariance_matrices[cluster_index],
            mean=means[cluster_index]
        )
        pdf = numpy.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])

        plt.contour(
            X, Y, pdf,
            colors=COLORS[cluster_index % len(COLORS)],
            alpha=float(probs[cluster_index])
        )

    plt.scatter(*data.T)
    plt.show()
