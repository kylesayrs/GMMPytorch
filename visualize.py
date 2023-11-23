import torch
import numpy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from torch.distributions.utils import logits_to_probs


COLORS = ["red", "blue", "green", "orange", "purple"]


def log_and_visualize(data: torch.Tensor, model: torch.nn.Module, loss: torch.Tensor):
    print(f"Loss: {loss.item():.2f}")
    plot_2d(data, model, loss)


def plot_2d(data: torch.Tensor, model: torch.nn.Module, loss: torch.Tensor):    
    probs = logits_to_probs(model.mixture.logits).detach()
    covariance_matrices = model.get_covariance_matrix().detach()
    means = model.mus.detach()

    for cluster_index in range(covariance_matrices.shape[0]):
        x = numpy.linspace(-20, 20, num=100)
        y = numpy.linspace(-20, 20, num=100)
        X, Y = numpy.meshgrid(x,y)

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
