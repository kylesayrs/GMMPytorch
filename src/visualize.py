import torch
import numpy
import warnings
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from src.utils import warp_probs


COLORS = ["red", "blue", "green", "orange", "purple"]


def plot_data_and_model(data: torch.Tensor, model: torch.nn.Module):
    if data.shape[1] == 2:
        plot_2d(data, model)

    else:
        warnings.warn("Visualization is not supported for the target data dimension")


def plot_2d(data: torch.Tensor, model: torch.nn.Module):    
    probs = model.get_probs().detach()
    covariance_matrices = model.get_covariance_matrix().detach()
    means = model.mus.detach()

    alphas = warp_probs(probs, 0.75)  # clearer visualization
    radius = int(torch.max(torch.abs(data)))
    X, Y = _make_mesh_grid(radius)
    for component_index in range(covariance_matrices.shape[0]):
        # create normal distribution
        distr = multivariate_normal(
            cov=covariance_matrices[component_index],
            mean=means[component_index]
        )

        # make pdf from distribution
        pdf = [
            [
                distr.pdf([X[i,j], Y[i,j]])
                for j in range(X.shape[1])
            ]
            for i in range(X.shape[0])
        ]

        # plot
        plt.contour(
            X, Y, pdf,
            colors=COLORS[component_index % len(COLORS)],
            alpha=float(alphas[component_index])
        )

    plt.gca().set_aspect("equal")
    plt.scatter(*data.T)
    plt.show()


def _make_mesh_grid(radius: int):
    x = numpy.linspace(-radius - 1, radius + 1, num=100)
    y = numpy.linspace(-radius - 1, radius + 1, num=100)
    return numpy.meshgrid(x, y)
