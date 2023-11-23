from typing import Tuple, List

import torch
import numpy

from FamilyTypes import MixtureFamily


def generate_data(
    num_samples: int,
    num_clusters: int,
    num_dims: int,
    width: float,
    family: MixtureFamily
) -> Tuple[List[numpy.ndarray], List[numpy.ndarray], torch.Tensor]:
    """
    Sample data from mock gaussian distributions

    :param num_samples: number of total samples
    :param num_clusters: number of mock gaussian distributions to sample from
    :param num_dims: number of dimensions
    :param width: width of possible means
    :return: true means, true covariance matrices, and samples
    """
    true_mus = []
    true_sigmas = []
    all_samples = []

    samples_per_cluster = num_samples // num_clusters

    for _ in range(num_clusters):
        true_mu = numpy.random.rand(num_dims) * width

        if family == MixtureFamily.FULL:
            true_sigma_sqrt = numpy.random.rand(num_dims, num_dims)
            true_sigma = true_sigma_sqrt.T @ true_sigma_sqrt
        else:
            true_sigma = numpy.diag(numpy.random.rand(num_dims))

        samples = numpy.random.multivariate_normal(true_mu, true_sigma, samples_per_cluster)

        true_mus.append(true_mu)
        true_sigmas.append(true_sigma)
        all_samples.append(samples)

    all_samples = numpy.concatenate(all_samples)
    all_samples = torch.tensor(all_samples, dtype=torch.float32)

    return all_samples, true_mus, true_sigmas
