from __future__ import annotations

from enum import StrEnum

from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.utils import MissingDict
from random_events.set import Set
from random_events.variable import Symbolic


# %% string-backed symbolic domain
class StringBackedCategory(StrEnum):
    """
    A symbolic domain whose members hash through Python's randomized string hash.
    """

    ALPHA = "ALPHA"
    BETA = "BETA"


def string_backed_distribution() -> SymbolicDistribution:
    """
    :return: A distribution over :class:`StringBackedCategory` with distinct
        probabilities per member, so a mix-up between members is visible.
    """
    variable = Symbolic(name="category", domain=Set.from_iterable(StringBackedCategory))
    probabilities = MissingDict(float)
    probabilities[hash(StringBackedCategory.ALPHA)] = 0.25
    probabilities[hash(StringBackedCategory.BETA)] = 0.75
    return SymbolicDistribution(variable=variable, probabilities=probabilities)
