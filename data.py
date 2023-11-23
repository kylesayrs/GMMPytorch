from typing import Tuple, List

import torch
import numpy

from FamilyTypes import MixtureFamily


def sample_data(
    num_samples: int,
    num_clusters: int,
    D: int,
    family: MixtureFamily
) -> Tuple[List[numpy.ndarray], List[numpy.ndarray], torch.Tensor]:
    """
    Sample data from mock gaussian distributions

    :param num_samples: number of total samples
    :param num_clusters: number of mock gaussian distributions to sample from
    :param D: number of dimensions
    :return: true means, true covariance matrices, and samples
    """
    true_mus = []
    true_sigmas = []
    all_samples = []

    samples_per_cluster = num_samples // num_clusters

    for _ in range(num_clusters):
        true_mu = numpy.random.rand(D) * 10

        if family == MixtureFamily.FULL:
            true_sigma_sqrt = numpy.random.rand(D, D)
            true_sigma = true_sigma_sqrt.T @ true_sigma_sqrt
        else:
            true_sigma = numpy.diag(numpy.random.rand(D))

        samples = numpy.random.multivariate_normal(true_mu, true_sigma, samples_per_cluster)

        true_mus.append(true_mu)
        true_sigmas.append(true_sigma)
        all_samples.append(samples)

    all_samples = numpy.concatenate(all_samples)
    all_samples = torch.tensor(all_samples, dtype=torch.float32)

    return all_samples, true_mus, true_sigmas
