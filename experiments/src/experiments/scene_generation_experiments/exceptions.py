from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


@dataclass
class ShelfLayoutResolutionError(DataclassException):
    """
    Raised when resolve_shelf_collisions cannot reach a collision-free, in-
    bounds layout within the allowed number of repair passes.
    """

    remaining_layer_indices: frozenset[int]
    """
    Indices into the layers list that still had a collision or an out-of-bounds
    object when resolution gave up.
    """

    passes_attempted: int
    """
    The number of repair passes attempted before giving up.
    """

    def error_message(self) -> str:
        return (
            f"Failed to resolve shelf layout after {self.passes_attempted} passes; "
            f"layers {sorted(self.remaining_layer_indices)} still have unresolved "
            "collisions or out-of-bounds objects."
        )

    def suggest_correction(self) -> str:
        return (
            "Re-sample the shelf from scratch, or check whether the sampled "
            "layer/object scales make a valid arrangement unreachable."
        )
