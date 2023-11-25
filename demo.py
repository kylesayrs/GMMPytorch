import torch
import numpy
import argparse
import warnings

from src.data import generate_data
from src.model import get_model
from src.FamilyTypes import FAMILY_NAMES, get_mixture_family_from_str
from src.fit_model import fit_model
from src.visualize import plot_data_and_model


parser = argparse.ArgumentParser(description="Fit a gaussian mixture model to generated mock data")
parser.add_argument("--samples", type=int, default=1000, help="The number of total samples in dataset")
parser.add_argument("--components", type=int, default=5, help="The number of gaussian components in mixture model")
parser.add_argument("--dims", type=int, default=2, help="The number of data dimensions")
parser.add_argument("--iterations", type=int, default=20_000, help="The number optimization steps")
parser.add_argument("--family", type=str, default="full", choices=FAMILY_NAMES, help="Model family, see `Mixture Types`")
parser.add_argument("--log_freq", type=int, default=5_000, help="Steps per log event")
parser.add_argument("--radius", type=float, default=8.0, help="L1 bound of data samples")
parser.add_argument("--mixture_lr", type=float, default=3e-5, help="Learning rate of mixture parameter (pi)")
parser.add_argument("--component_lr", type=float, default=1e-2, help="Learning rate of component parameters (mus, sigmas)")
parser.add_argument("--visualize", type=bool, default=True, help="True for visualization at each log event and end")
parser.add_argument("--seed", type=int, default=0, help="seed for numpy and torch")


if __name__ == "__main__":
    args = parser.parse_args()
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # priors used for both generation and modelling
    mixture_family = get_mixture_family_from_str(args.family)
    data_radius = args.radius

    # load data
    data, true_mus, true_sigmas = generate_data(
        args.samples,
        args.components,
        args.dims,
        data_radius,
        mixture_family,
        args.seed
    )

    # set up model
    model = get_model(
        mixture_family,
        args.components,
        args.dims,
        data_radius
    )

    # fit model
    loss = fit_model(
        model,
        data,
        args.iterations,
        args.mixture_lr,
        args.component_lr,
        args.log_freq,
        args.visualize
    )

    # visualize
    print(f"Final Loss: {loss:.2f}")
    if args.visualize:
        plot_data_and_model(data, model)
