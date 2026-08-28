from __future__ import annotations

import math
from dataclasses import dataclass

from krrood.entity_query_language.factories import a


@dataclass
class _EagerlyComputingPoint:
    """
    Mimics a factory that computes something from its arguments in ``__init__``
    (like :class:`~semantic_digital_twin.spatial_types.spatial_types.Pose2D`'s eager
    conversion to a casadi symbolic vector), rather than merely storing them.
    """

    x: float
    y: float

    def __post_init__(self):
        self.magnitude = math.hypot(self.x, self.y)


@dataclass
class _NamedPoint:
    name: str
    point: _EagerlyComputingPoint


@dataclass
class _NamedPoints:
    points: list


# %% constructing a query holding an unresolved nested attribute


def test_construct_instance_leaves_an_unresolved_nested_attribute_as_none() -> None:
    """
    A query built through :func:`a` for grounding (see
    ``RelationalProbabilisticCircuit.ground``) may hold entries whose own attributes
    are still ``...``, left for a model to answer later -- alongside entries that are
    already fully known. Constructing the *whole* query eagerly must not try to call
    an unresolved entry's own factory with ``...`` as a real argument: that only
    plain field-storing constructors tolerate, and this dataclass's
    ``__post_init__`` -- like ``Pose2D.__init__`` -- computes something from its
    arguments immediately and cannot. The unresolved entry's ``point`` is left
    ``None`` instead; nothing else about it is touched, in particular its ``name``,
    which is not the field left unresolved.
    """
    query = a(_NamedPoints)(
        points=[
            a(_NamedPoint)(
                name="fixed", point=_EagerlyComputingPoint(x=1.0, y=2.0)
            ),
            a(_NamedPoint)(name="free", point=a(_EagerlyComputingPoint)(x=..., y=...)),
        ]
    )

    instance = query.construct_instance()

    assert len(instance.points) == 2
    assert instance.points[0].point.magnitude == math.hypot(1.0, 2.0)
    assert instance.points[1].name == "free"
    assert instance.points[1].point is None
