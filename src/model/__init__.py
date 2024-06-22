import torch

from .full import GmmFull
from .diagonal import GmmDiagonal
from .isotropic import GmmIsotropic
from .shared import GmmSharedIsotropic
from src.FamilyTypes import MixtureFamily


def get_model(
    mixture_family: MixtureFamily,
    num_components: int,
    num_dims: int,
    radius: float
) -> torch.nn.Module:
    if mixture_family == MixtureFamily.FULL:
        return GmmFull(num_components, num_dims, radius)
    
    if mixture_family == MixtureFamily.DIAGONAL:
        return GmmDiagonal(num_components, num_dims, radius)
    
    if mixture_family == MixtureFamily.ISOTROPIC:
        return GmmIsotropic(num_components, num_dims, radius)
    
    if mixture_family == MixtureFamily.SHARED_ISOTROPIC:
        return GmmSharedIsotropic(num_components, num_dims, radius)

    raise NotImplementedError(
        f"Mixture family {mixture_family.value} not implemented yet"
    )
