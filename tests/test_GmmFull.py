import pytest
import torch
import numpy

from src.model import GmmFull
from src.fit_model import fit_model


SEED = 42
@pytest.fixture(scope="function")
def seed():
    numpy.random.seed(SEED)
    torch.manual_seed(SEED)


IDENTITY_1d = [[1]]
HIGH_VARIANCE_1d = [[10]]
IDENTITY_2d = [
    [1, 0],
    [0, 1]
]
DIAG_LOW_VARIANCE_2d = [
    [0.5, 0],
    [0, 0.5]
]
DIAG_HIGH_VARIANCE_2d = [
    [10, 0],
    [0, 10]
]
FIRST_HIGH_VARIANCE_2d = [
    [10, 0],
    [0, 0.5]
]
SECOND_HIGH_VARIANCE_2d = [
    [0.5, 0],
    [0, 10]
]
POS_CORRELATION_2d = [
    [4, 3],
    [3, 8]
]
NEG_CORRELATION_2d = [
    [4, -3],
    [-3, 8]
]
MIX_5d = [
    [ 1.   , -0.392,  0.188,  0.53 ,  0.331],
    [-0.392,  1.   ,  0.395,  0.044,  0.044],
    [ 0.188,  0.395,  1.   ,  0.251,  0.569],
    [ 0.53 ,  0.044,  0.251,  1.   ,  0.662],
    [ 0.331,  0.044,  0.569,  0.662,  1.   ]
]
IDENTITY_5d = [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
]
@pytest.mark.parametrize(
    "mean,sigma,exp_loss",
    [
        ([0], IDENTITY_1d, 1.3),
        ([0], HIGH_VARIANCE_1d, 2.5),
        ([0, 0], IDENTITY_2d, 2.6),
        ([0, 0], DIAG_LOW_VARIANCE_2d, 2.1),
        ([0, 0], DIAG_HIGH_VARIANCE_2d, 4.9),
        ([0, 0], FIRST_HIGH_VARIANCE_2d, 3.5),
        ([0, 0], SECOND_HIGH_VARIANCE_2d, 3.5),
        ([0, 0], POS_CORRELATION_2d, 4.1),
        ([0, 0], NEG_CORRELATION_2d, 4.1),
        ([0, 0, 0, 0, 0], IDENTITY_5d, 6.7),
        ([0, 0, 0, 0, 0], MIX_5d, 5.7),
    ],
)
def test_GmmFull_covariances(mean, sigma, exp_loss, seed):
    samples_per_cluster = 200
    num_iterations = 5_000
    mixture_lr = 0.0
    component_lr = 0.05
    num_components = 1
    num_dims = len(sigma[0])
    
    data = numpy.random.multivariate_normal(mean, sigma, samples_per_cluster)
    data = torch.tensor(data)

    init_radius = numpy.max(numpy.abs(data.numpy()), axis=None)
    model = GmmFull(num_components, num_dims, init_radius)

    loss = fit_model(
        model,
        data,
        num_iterations,
        mixture_lr,
        component_lr,
        visualize=False
    )

    assert loss == pytest.approx(exp_loss, 0.1)
    assert torch.dist(model.mus[0], torch.tensor(mean)) < 0.15


@pytest.mark.parametrize(
    "mean,sigma,exp_loss",
    [
        ([-1], IDENTITY_1d, 1.3),
        ([ 1], IDENTITY_1d, 1.3),

        ([-1, -1], IDENTITY_2d, 2.6),
        ([ 1,  1], IDENTITY_2d, 2.6),
        ([-1,  1], IDENTITY_2d, 2.6),
        ([ 1, -1], IDENTITY_2d, 2.6),

        ([0, 0, 0, 0, 0], IDENTITY_5d, 7.0),
        ([0, -1, 1, 0, -1], IDENTITY_5d, 7.0),
        ([-1, 0, 0, 1, 0], IDENTITY_5d, 7.0),
        ([1, 1, -1, -1, 1], IDENTITY_5d, 7.0),
    ],
)
def test_GmmFull_means(mean, sigma, exp_loss, seed):
    samples_per_cluster = 200
    num_iterations = 5_000
    mixture_lr = 0.0
    component_lr = 0.1
    num_components = 1
    num_dims = len(sigma)
    
    data = numpy.random.multivariate_normal(mean, sigma, samples_per_cluster)
    data = torch.tensor(data)

    init_radius = numpy.max(numpy.abs(data.numpy()), axis=None)
    model = GmmFull(num_components, num_dims, init_radius)

    loss = fit_model(
        model,
        data,
        num_iterations,
        mixture_lr,
        component_lr,
        visualize=False
    )

    assert loss == pytest.approx(exp_loss, 0.1)
    assert torch.dist(model.mus[0], torch.tensor(mean)) < 0.15


@pytest.mark.parametrize(
    "means,sigmas,exp_loss",
    [
        (
            [
                [-6],
                [-3],
                [ 3],
                [ 6]
            ],
            [
                IDENTITY_1d,
                IDENTITY_1d,
                IDENTITY_1d,
                IDENTITY_1d
            ],
            2.6
        ),
        (
            [
                [-6],
                [-3],
                [ 3],
                [ 6]
            ],
            [
                HIGH_VARIANCE_1d,
                HIGH_VARIANCE_1d,
                HIGH_VARIANCE_1d,
                HIGH_VARIANCE_1d
            ],
            3.1
        ),
        (
            [
                [-10, -10],
                [ 10, -10],
                [-10,  10],
                [ 10,  10]
            ],
            [
                IDENTITY_2d,
                IDENTITY_2d,
                IDENTITY_2d,
                IDENTITY_2d
            ],
            4.2
        ),
        (
            [
                [-1, -1],
                [ 1, -1],
                [-1,  1],
                [ 1,  1]
            ],
            [
                DIAG_HIGH_VARIANCE_2d,
                DIAG_LOW_VARIANCE_2d,
                POS_CORRELATION_2d,
                NEG_CORRELATION_2d
            ],
            4.2
        ),
        (
            [
                [ 10,  0, 0, 0, 0],
                [  0, 10, 0, 0, 0],
            ],
            [
                IDENTITY_5d,
                IDENTITY_5d,
            ],
            7.3
        ),
        (
            [
                [ 10, 0, 0, 0, 0],
                [-10, 0, 0, 0, 0],
            ],
            [
                MIX_5d,
                MIX_5d,
            ],
            6.1
        ),
    ],
)
def test_GmmFull_components(means, sigmas, exp_loss, seed):
    samples_per_cluster = 200
    num_iterations = 5_000
    mixture_lr = 0.0
    component_lr = 0.1
    num_components = len(sigmas)
    num_dims = len(sigmas[0])
    
    data = numpy.concatenate([
        numpy.random.multivariate_normal(mean, sigma, samples_per_cluster)
        for mean, sigma in zip(means, sigmas)
    ])
    data = torch.tensor(data)

    init_radius = numpy.max(numpy.abs(data.numpy()), axis=None)
    model = GmmFull(num_components, num_dims, init_radius)

    loss = fit_model(
        model,
        data,
        num_iterations,
        mixture_lr,
        component_lr,
        visualize=False
    )

    assert loss == pytest.approx(exp_loss, 0.1)
    # cannot test means due to identifiability issue


@pytest.mark.parametrize(
    "data,init_mus,exp_loss",
    [
        ([[0.0]],        [[0.0]],        1.1),
        ([[0.0], [1.0]], [[0.0], [1.0]], 0.7),
    ],
)
def test_GmmFull_Singularity(data, init_mus, exp_loss, seed):
    num_iterations = 5_000
    mixture_lr = 0.0
    component_lr = 0.1
    num_components = 2
    init_radius = 1.0
    num_dims = len(data[0])
    
    data = torch.tensor(data)

    model = GmmFull(num_components, num_dims, init_radius, init_mus)

    loss = fit_model(
        model,
        data,
        num_iterations,
        mixture_lr,
        component_lr,
        visualize=False
    )

    assert loss == pytest.approx(exp_loss, 0.1)
