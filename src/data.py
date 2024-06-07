from typing import Tuple, List

import torch
import numpy

from src.FamilyTypes import MixtureFamily
from src.utils import make_random_cov_matrix


def generate_data(
    num_samples: int,
    num_clusters: int,
    num_dims: int,
    radius: float,
    family: MixtureFamily
) -> Tuple[List[numpy.ndarray], List[numpy.ndarray], torch.Tensor]:
    """
    Sample data from mock gaussian distributions

    :param num_samples: number of total samples
    :param num_clusters: number of mock gaussian distributions to sample from
    :param num_dims: number of dimensions
    :param radius: l1 radius of possible means
    :param family: distribution family used for both modeling and data generating
    :return: true means, true covariance matrices, and samples
    """
    true_mus = []
    true_sigmas = []
    all_samples = []

    samples_per_cluster = num_samples // num_clusters

    shared_random_sigma = numpy.random.random(1)
    for _ in range(num_clusters):
        true_mu = numpy.random.uniform(-radius, radius, num_dims)

        if family == MixtureFamily.FULL:
            true_sigma = make_random_cov_matrix(num_dims)
        elif family == MixtureFamily.DIAGONAL:
            true_sigma = numpy.diag(numpy.random.random(num_dims))
        elif family == MixtureFamily.ISOTROPIC:
            true_sigma = numpy.diag(numpy.random.random(1).repeat(num_dims))
        elif family == MixtureFamily.SHARED_ISOTROPIC:
            true_sigma = numpy.diag(shared_random_sigma.repeat(num_dims))
        else:
            true_sigma = numpy.diag([1] * num_dims)

        samples = numpy.random.multivariate_normal(true_mu, true_sigma, samples_per_cluster)

        true_mus.append(true_mu)
        true_sigmas.append(true_sigma)
        all_samples.append(samples)

    all_samples = numpy.concatenate(all_samples)
    all_samples = torch.tensor(all_samples, dtype=torch.float32)

    return all_samples, true_mus, true_sigmas
