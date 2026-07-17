from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


@dataclass
class LayoutResolutionError(DataclassException):
    """
    Raised when the in-world resolver cannot reach a collision-free, supported
    layout within the allowed number of repair passes.
    """

    remaining_groups: frozenset[int]
    """
    Indices of the collision groups that still had a colliding or unsupported
    object when resolution gave up.
    """

    passes_attempted: int
    """
    The number of repair passes attempted before giving up.
    """

    def error_message(self) -> str:
        return (
            f"Failed to resolve layout after {self.passes_attempted} passes; "
            f"groups {sorted(self.remaining_groups)} still have unresolved "
            "collisions or unsupported objects."
        )

    def suggest_correction(self) -> str:
        return (
            "Re-sample the layout from scratch, or check whether the sampled "
            "scales make a valid arrangement unreachable."
        )
