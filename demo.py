import torch
import numpy
import argparse
import warnings

from src.data import generate_data
from src.model import get_model
from src.visualize import log_and_visualize
from src.FamilyTypes import FAMILY_NAMES, get_mixture_family_from_str


parser = argparse.ArgumentParser()
parser.add_argument("--samples", default=500)
parser.add_argument("--clusters", default=5)
parser.add_argument("--mixtures", default=5)
parser.add_argument("--dims", default=2)
parser.add_argument("--iterations", default=10_000)
parser.add_argument("--family", type=str, default="full", choices=FAMILY_NAMES)
parser.add_argument("--log_freq", type=int, default=3_000)
parser.add_argument("--width", type=float, default=5.0)
parser.add_argument("--mixture_lr", type=float, default=3e-5)
parser.add_argument("--component_lr", type=float, default=1e-2)
parser.add_argument("--seed", type=int, default=402)


if __name__ == "__main__":
    args = parser.parse_args()
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # warnings
    if args.dims != 2:
        warnings.warn("Visualization is only supported for dims=2")
    
    # use mixture family for both generation and modelling
    mixture_family = get_mixture_family_from_str(args.family)

    # load data
    data, true_mus, true_sigmas = generate_data(
        args.samples, args.clusters, args.dims, args.width, mixture_family
    )

    # set up model
    model = get_model(mixture_family, args.mixtures, args.dims, args.width)

    # create separate optimizers for mixture coeficients and components
    mixture_optimizer = torch.optim.Adam(model.mixture_parameters(), lr=args.mixture_lr)
    mixture_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mixture_optimizer, args.iterations)
    components_optimizer = torch.optim.Adam(model.component_parameters(), lr=args.component_lr)
    components_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(components_optimizer, args.iterations)

    # optimize
    for iteration_index in range(args.iterations):
        # reset gradient
        components_optimizer.zero_grad()
        mixture_optimizer.zero_grad()

        # forward
        loss = model(data)

        # log and visualize
        if iteration_index % args.log_freq == 0:
            log_and_visualize(iteration_index, data, model, loss, args.width)

        # backwards
        loss.backward()
        mixture_optimizer.step()
        mixture_scheduler.step()
        components_optimizer.step()
        components_scheduler.step()
    
    log_and_visualize(iteration_index, data, model, loss, args.width)
