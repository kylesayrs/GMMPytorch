from enum import Enum


class MixtureFamily(Enum):
    FULL = "full"                          # fully expressive eigenvalues
    DIAGONAL = "diagonal"                  # eigenvalues align with data axes
    ISOTROPIC = "isotropic"                # same variance for all directions
    SHARED_ISOTROPIC = "shared_isotropic"  # same variance for all directions and components
    CONSTANT = "constant"                  # sigma is not learned


FAMILY_NAMES = [family.value for family in MixtureFamily]


def get_mixture_family_from_str(family_name: str):
    for family in MixtureFamily:
        if family.value == family_name:
            return family

    raise ValueError(
        f"Unknown mixture family `{family_name}`. "
        f"Please select from {FAMILY_NAMES}"
    )
