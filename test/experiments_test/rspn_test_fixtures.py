"""
A tiny, fully factorized shelf RSPN for testing -- no learned correlations, cheap to
fit, usable anywhere a real :class:`RelationalProbabilisticCircuit` is required
instead of a :class:`~unittest.mock.MagicMock`.

:func:`build_factorized_shelf_rspn` exercises the real fit/ground/sample pipeline
(:mod:`krrood.entity_query_language.backends`,
:class:`~probabilistic_model.probabilistic_circuit.relational.rspn.RelationalProbabilisticCircuit`)
end to end, unlike patching
:func:`~experiments.scene_generation_experiments.rspn_sampling.probabilistic_backend`
to hand back a canned :class:`EGShelfLayer` -- so it can stand in for the RSPN in
:class:`~experiments.scene_generation_experiments.in_world_resolver.InWorldLayoutResolver`
tests that want to exercise the actual resample query instead of a hard-coded one.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from krrood.ormatic.data_access_objects.helper import to_dao
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProductUnit,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGShelf,
    EGShelfLayer,
    ObjectType,
)
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world_description.geometry import Scale

FULLY_FACTORIZED_MIN_SAMPLES_PER_LEAF = 0.99
"""
``min_samples_per_leaf`` fraction that keeps a fitted circuit to a single leaf at
every level.

A JPT only splits a node into two children when each child would still meet
``min_samples_per_leaf``; a fraction above one half can never be met by two children
of the same parent, whatever the row count or how the rows are distributed, so the
fit stops at one leaf per level (see
:meth:`~probabilistic_model.learning.jpt.jpt.JointProbabilityTree.fit`, which checks
this before ever attempting a split). A JPT leaf is a
:class:`~probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit.ProductUnit`
of independent per-variable distributions -- the correlations a JPT can represent all
come from mixing *multiple* leaves, never from a single one -- so stopping at one leaf
is what makes the circuit fully factorized rather than merely close to it, regardless
of how the training rows were generated.
"""


def _random_object(
    rng: random.Random,
    object_type: ObjectType,
    index: int,
    position_extent: float,
    scale_range: tuple[float, float],
) -> EGObject2D:
    """
    An object with independently drawn scale, position and yaw, so none of its own
    fields carry any relationship to each other or to any other object's fields.
    """
    return EGObject2D(
        object_type=object_type,
        scale=Scale(
            x=rng.uniform(*scale_range),
            y=rng.uniform(*scale_range),
            z=rng.uniform(*scale_range),
        ),
        pose=Pose2D(
            x=rng.uniform(-position_extent, position_extent),
            y=rng.uniform(-position_extent, position_extent),
            yaw=rng.uniform(-3.14, 3.14),
        ),
        source_id=f"{object_type.value}_{index}",
    )


def _random_layer(
    rng: random.Random,
    theme: ObjectType,
    object_types: Sequence[ObjectType],
    relative_height: float,
    shelf_height: float,
    position_extent: float,
    scale_range: tuple[float, float],
) -> EGShelfLayer:
    return EGShelfLayer(
        objects=[
            _random_object(rng, object_type, index, position_extent, scale_range)
            for index, object_type in enumerate(object_types)
        ],
        theme_dominant_type=theme,
        height_above_shelf_base=relative_height * shelf_height,
        relative_height=relative_height,
        vertical_clearance=rng.uniform(0.2, 0.4),
    )


def _random_shelf(
    rng: random.Random,
    theme: ObjectType,
    object_types: Sequence[ObjectType],
    layers_per_shelf: int,
    objects_per_layer: int,
    shelf_height: float,
    position_extent: float,
    scale_range: tuple[float, float],
) -> EGShelf:
    shelf_extent = 2 * position_extent + 2 * scale_range[1]
    return EGShelf(
        scale=Scale(
            x=rng.uniform(0.8, 1.0) * shelf_extent,
            y=rng.uniform(0.8, 1.0) * shelf_extent,
            z=shelf_height,
        ),
        layers=[
            _random_layer(
                rng,
                theme,
                [
                    object_types[(layer_index + slot) % len(object_types)]
                    for slot in range(objects_per_layer)
                ],
                relative_height=(layer_index + 1) / (layers_per_shelf + 1),
                shelf_height=shelf_height,
                position_extent=position_extent,
                scale_range=scale_range,
            )
            for layer_index in range(layers_per_shelf)
        ],
        theme_dominant_type=theme,
    )


def build_factorized_shelf_rspn(
    themes: Sequence[ObjectType] = (ObjectType.BOOK, ObjectType.BOTTLE),
    object_types: Sequence[ObjectType] = (
        ObjectType.BOOK,
        ObjectType.BOTTLE,
        ObjectType.BOX,
    ),
    shelves_per_theme: int = 4,
    layers_per_shelf: int = 2,
    objects_per_layer: int = 2,
    shelf_height: float = 1.2,
    position_extent: float = 0.3,
    scale_range: tuple[float, float] = (0.05, 0.25),
    seed: int = 0,
) -> RelationalProbabilisticCircuit:
    """
    Fit a small :class:`EGShelf` RSPN whose scale, pose and every other continuous
    field is modelled fully independently at every relational level (shelf, layers,
    objects) -- no field's fitted distribution depends on any other field's value.

    Because every level of the fit stops at a single leaf (see
    :data:`FULLY_FACTORIZED_MIN_SAMPLES_PER_LEAF`), the randomness the training rows
    are drawn with only decides how spread out each field's own marginal is -- never
    whether two fields end up correlated. A test can therefore condition this circuit
    on one field and still draw independently varied samples for the others, which is
    what makes it useful as a cheap stand-in for a real RSPN wherever a test wants to
    exercise the actual query/grounding/sampling path (see module docstring) rather
    than mock it out.

    :param themes: Shelf themes to draw training shelves for; each theme gets its own
        block of *shelves_per_theme* shelves.
    :param object_types: Pool of object types layers draw their objects from.
    :param shelves_per_theme: Training shelves fit per theme.
    :param layers_per_shelf: Layers each training shelf has.
    :param objects_per_layer: Objects each training layer holds.
    :param shelf_height: Corpus height every training shelf shares.
    :param position_extent: Half-width, in metres, of the square every training
        object's x/y is drawn from. A query conditioning a fixed object's pose outside
        this range -- or a free-space truncation that only overlaps positions outside
        it -- has zero density here and raises ``NoSolutionFound``; widen this to match
        the scale of whatever real geometry (e.g. a mesh much bigger than
        *scale_range*) a caller spawns objects with.
    :param scale_range: ``(min, max)`` every training object's x/y/z scale is drawn
        from independently. Conditioning on a declared scale outside this range is
        the other way a query ends up with zero density.
    :param seed: Seed for the training data's random draws; the same seed always
        produces the same fitted circuit.
    :return: The fitted, fully factorized circuit.
    """
    rng = random.Random(seed)
    shelves = [
        _random_shelf(
            rng,
            theme,
            object_types,
            layers_per_shelf,
            objects_per_layer,
            shelf_height,
            position_extent,
            scale_range,
        )
        for theme in themes
        for _ in range(shelves_per_theme)
    ]
    return RelationalProbabilisticCircuit(
        EGShelf, min_samples_per_leaf=FULLY_FACTORIZED_MIN_SAMPLES_PER_LEAF
    ).fit([to_dao(shelf) for shelf in shelves])


def is_single_leaf(circuit: RelationalProbabilisticCircuit) -> bool:
    """
    Whether *circuit* -- and every nested exchangeable-part template it owns -- fit to
    exactly one leaf, i.e. is a plain product distribution with no learned
    correlations anywhere in it.

    :param circuit: The fitted circuit to check, recursively.
    :return: ``True`` if *circuit* and all of its exchangeable-part templates are
        single-leaf.
    """
    root_children = circuit.class_probabilistic_circuit.root.subcircuits
    if len(root_children) != 1 or not isinstance(root_children[0], ProductUnit):
        return False
    return all(
        is_single_leaf(template.template_distribution)
        for template in circuit.exchangeable_distribution_templates.values()
    )


@pytest.fixture
def factorized_shelf_rspn() -> RelationalProbabilisticCircuit:
    """
    A cheap, fully factorized :class:`EGShelf` circuit for tests that need a real RSPN
    to sample from without paying for (or depending on the correlations learned by) a
    circuit fitted on real shelf data.
    """
    return build_factorized_shelf_rspn()
