from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


@dataclass
class OutdatedTrainedModelError(DataclassException):
    """
    Raised when a cached model was fitted before the schema it is loaded against.

    A model that no longer matches the schema still loads and still samples, so
    without this the demo would quietly serve shelves drawn from a distribution
    that knows nothing of what it was asked for.
    """

    model_path: str
    """
    File the outdated model was read from.
    """

    missing_variable: str
    """
    Variable the schema expects the fitted circuit to model, but which it does not.
    """

    def error_message(self) -> str:
        return (
            f"The model cached at {self.model_path} does not model "
            f"{self.missing_variable!r}, so it was fitted before that field existed."
        )

    def suggest_correction(self) -> str:
        return (
            "Delete the cached model so it is refitted from the processed "
            "database on the next run."
        )


@dataclass
class UnknownShelfVariableError(DataclassException):
    """
    Raised when a fitted circuit does not model a variable the sampler needs.

    Drawing a shelf's dimensions means reading named variables off the circuit, so
    a name that resolves to nothing would otherwise surface far downstream as an
    empty draw rather than as the fit being unusable.
    """

    variable_name: str
    """
    Name the sampler looked for.
    """

    modelled_variables: list[str]
    """
    Names the circuit does model, so the mismatch can be seen at a glance.
    """

    def error_message(self) -> str:
        return (
            f"The fitted circuit does not model {self.variable_name!r}. "
            f"It models: {', '.join(self.modelled_variables)}."
        )

    def suggest_correction(self) -> str:
        return (
            "Refit the model against the current schema; a circuit fitted before "
            "a field existed cannot be conditioned on it."
        )


@dataclass
class UndrawableShelfError(DataclassException):
    """
    Raised when no shelf of the requested theme could be drawn.

    Every attempt asked the circuit for a shelf it gives no probability to, which
    means the fit has nothing to say about that combination rather than that the
    draw was unlucky.
    """

    requested_theme: str
    """
    The dominant object type that was asked for.
    """

    requested_layer_count: Optional[int]
    """
    Layer count the caller pinned, or ``None`` when it was left to the model.
    """

    attempts: int
    """
    How many draws were tried.
    """

    def error_message(self) -> str:
        pinned = (
            f" with {self.requested_layer_count} layers"
            if self.requested_layer_count is not None
            else ""
        )
        return (
            f"No {self.requested_theme}-dominant shelf{pinned} could be drawn in "
            f"{self.attempts} attempts; the fitted model gives that shelf no "
            "probability."
        )

    def suggest_correction(self) -> str:
        return (
            "Leave the layer count to the model, or refit on data that contains "
            "shelves of this theme and size."
        )
