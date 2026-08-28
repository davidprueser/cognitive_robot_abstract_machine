from __future__ import annotations

import math

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from experiments.scene_generation_experiments.shelf_placement import (
    _layer_query,
    mode_query,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGShelf,
    EGShelfLayer,
    ObjectType,
)
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world_description.geometry import Scale

_BOOK_SCALE = Scale(x=0.12, y=0.2, z=0.04)
"""
Size shared by every book in this module, so the held object's pinned scale matches what
the circuit was fitted on.
"""

_SHELF_SCALE = Scale(x=0.4, y=0.8, z=1.2)


def _shelf_book(source_id: str) -> EGObject2D:
    return EGObject2D(
        object_type=ObjectType.BOOK,
        scale=_BOOK_SCALE,
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        source_id=source_id,
    )


def _single_layer_shelf(index: int) -> EGShelf:
    """
    A shelf with one layer already holding one book -- the minimal shape a
    :func:`~experiments.scene_generation_experiments.shelf_placement.mode_query` call
    needs to score, since it must place a second, held book alongside it.
    """
    layer = EGShelfLayer(
        objects=[_shelf_book(f"existing_book_{index}")],
        theme_dominant_type=ObjectType.BOOK,
        height_above_shelf_base=0.36,
        relative_height=0.3,
        vertical_clearance=0.3,
    )
    return EGShelf(
        scale=_SHELF_SCALE, layers=[layer], theme_dominant_type=ObjectType.BOOK
    )


# %% mode_query grounding with an underspecified held-object pose


def test_mode_query_places_a_held_object_with_a_free_pose() -> None:
    """
    Regression test: ``mode_query`` builds its held-object query slot with
    ``pose=a(Pose2D)(x=..., y=..., yaw=...)`` (:func:`_held_object_slot` in
    ``shelf_placement.py``), leaving every pose field underspecified so the circuit's
    mode search answers it.

    ``RelationalProbabilisticCircuit.ground`` calls ``query.construct_instance()`` to
    build a concrete instance for aggregation statistics before the mode search ever
    runs. That eagerly constructs every nested field, including the held slot's
    ``Pose2D`` -- and ``Pose2D.__init__`` converts its arguments to a casadi symbolic
    vector immediately, which cannot accept the ``Ellipsis`` placeholder an
    underspecified field carries. Previously (``EGObject2D.position: EGPoint2D`` /
    ``.orientation: EGRotation``, both plain dataclasses with no ``__init__`` logic)
    the same placeholder survived construction untouched; ``Pose2D`` does not tolerate
    it, so every ``mode_query`` call currently fails before it can answer at all.
    """
    shelf = _single_layer_shelf(index=0)
    shelf.spawn()
    circuit = RelationalProbabilisticCircuit(EGShelf, min_samples_per_leaf=0.1).fit(
        [to_dao(_single_layer_shelf(index)) for index in range(10)]
    )
    held_book = EGObject2D(
        object_type=ObjectType.BOOK,
        scale=_BOOK_SCALE,
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        source_id="held_book",
    )

    placed_object, layer_name = mode_query(shelf, circuit, held_book)

    assert layer_name == str(shelf.layers[0].annotation.root.name)
    assert placed_object.object_type == ObjectType.BOOK
    assert placed_object.scale == _BOOK_SCALE
    assert math.isfinite(float(placed_object.pose.x))
    assert math.isfinite(float(placed_object.pose.y))
    assert math.isfinite(float(placed_object.pose.yaw))


# %% _layer_query where-condition wiring


def test_layer_query_where_condition_names_the_query_s_own_pose_variables() -> None:
    """
    Regression test: ``_layer_query`` used to build its where-condition from
    ``held_slot.variable.pose`` while ``held_slot`` (an :class:`EGObject2D` match) was
    still unresolved, so ``held_slot.variable`` was its own standalone subject variable
    rather than the query's "objects[N]" access-path variable it becomes once the
    ``EGShelfLayer`` match resolves its ``objects`` list. The where-condition ended up
    naming a variable the fitted circuit never has, so the free-space truncation it
    exists for silently never took effect.
    """
    shelf = _single_layer_shelf(index=0)
    shelf.spawn()
    layer = shelf.layers[0]
    held_book = EGObject2D(
        object_type=ObjectType.BOOK,
        scale=_BOOK_SCALE,
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        source_id="held_book",
    )

    query = _layer_query(layer, held_book)
    parameters = UnderspecifiedParameters(query)

    where_condition_names = {
        variable.name
        for variable in parameters.truncation_assignments_from_where_conditions.variables
    }
    held_slot_index = len(layer.objects)
    assert where_condition_names == {
        f"EGShelfLayer.objects[{held_slot_index}].pose.x",
        f"EGShelfLayer.objects[{held_slot_index}].pose.y",
    }
    assert where_condition_names <= set(parameters.variables)
