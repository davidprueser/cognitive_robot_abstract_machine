from __future__ import annotations

import dataclasses

import pytest

from experiments.shelf_generation_experiments.placement.shelf_placement import (
    Placement,
)
from experiments.shelf_generation_experiments.tidying_demo.exceptions import (
    UnreachableShelfError,
)
from experiments.shelf_generation_experiments.tidying_demo.shelf_tidying import (
    FloorNavigation,
    ShelfFront,
)
from experiments.shelf_generation_experiments.utils import ObjectType
from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentObject2D,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Floor,
    Table,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose2D,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

from ..shelf_dataset import random_shelves


# %% helpers
def _floor_with_table(table_scale: Scale) -> tuple[World, Floor]:
    """
    :return: A world whose four by four metre floor has a table standing at its centre.
    """
    world = World.create_with_root_body()
    with world.modify_world():
        floor = Floor.create_with_new_body_in_world(
            name="floor",
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-0.01),
            scale=Scale(x=4.0, y=4.0, z=0.02),
        )
        table = Table.create_with_new_body_in_world(
            name="table",
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=table_scale.z / 2
            ),
            scale=table_scale,
        )
        floor.calculate_supporting_surface()
        floor.add_object(table)
    return world, floor


# %% floor navigation
def test_a_point_is_projected_onto_the_floor_plane() -> None:
    world, floor = _floor_with_table(Scale(x=0.5, y=0.5, z=0.5))

    point = FloorNavigation(floor).point_on_floor(
        Point3(1.0, -1.0, 1.5, reference_frame=world.root)
    )

    in_world = world.transform(point.to_point3(), world.root)
    assert point.reference_frame is floor.root
    assert [float(in_world.x), float(in_world.y)] == pytest.approx([1.0, -1.0])


def test_a_route_around_the_table_leads_through_free_floor() -> None:
    world, floor = _floor_with_table(Scale(x=0.5, y=2.0, z=0.5))
    navigation = FloorNavigation(floor)

    goals = navigation.route(
        Point3(-1.5, 0.0, 0.0, reference_frame=world.root),
        Point3(1.5, 0.0, 0.0, reference_frame=world.root),
    )

    assert goals
    for goal in goals:
        assert (
            navigation.free_space.node_of_point(
                navigation.point_on_floor(goal.position)
            )
            is not None
        )


def test_a_straight_way_needs_no_goals() -> None:
    world, floor = _floor_with_table(Scale(x=0.5, y=0.5, z=0.5))

    goals = FloorNavigation(floor).route(
        Point3(-1.5, 1.5, 0.0, reference_frame=world.root),
        Point3(1.5, 1.5, 0.0, reference_frame=world.root),
    )

    assert goals == []


def test_a_table_across_the_whole_floor_leaves_no_route() -> None:
    world, floor = _floor_with_table(Scale(x=0.5, y=4.0, z=0.5))

    with pytest.raises(UnreachableShelfError):
        FloorNavigation(floor).route(
            Point3(-1.5, 0.0, 0.0, reference_frame=world.root),
            Point3(1.5, 0.0, 0.0, reference_frame=world.root),
        )


# %% shelf front
def test_the_robot_stands_in_front_of_the_open_face_level_with_the_placement() -> None:
    shelf = dataclasses.replace(random_shelves()[0], source_ids=[])
    shelf.spawn(World.create_with_root_body(), corpus_wall_thickness=0.03)
    front = ShelfFront(shelf=shelf, corpus_wall_thickness=0.03, standoff=0.5)
    placement = Placement(
        placed_object=RelationalCircuitExperimentObject2D(
            object_type=ObjectType.BOOK,
            scale=Scale(x=0.1, y=0.1, z=0.1),
            pose=Pose2D(x=0.1, y=0.2),
            source_id="book",
        ),
        layer=shelf.layers[0],
        log_density=0.0,
    )

    standing_point = front.standing_point(placement)

    expected_offset = shelf.annotation.hole_direction * (
        shelf.corpus_footprint(0.03).x / 2 + 0.5
    )
    assert standing_point.reference_frame is shelf.corpus
    assert [float(standing_point.x), float(standing_point.y)] == pytest.approx(
        [float(expected_offset.x), 0.2]
    )
