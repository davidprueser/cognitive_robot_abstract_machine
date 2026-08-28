from __future__ import annotations

from pathlib import Path

import pytest

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from experiments.scene_generation_experiments.utils import rclpy_node
from krrood.ormatic.data_access_objects.helper import to_dao
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from experiments.scene_generation_experiments.rspn_sampling import (
    free_object_slot,
    build_layer_query,
    build_theme_shelf_query,
    probabilistic_backend,
)
from experiments.scene_generation_experiments.shelf_placement import (
    layer_named,
    mode_query,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGShelf,
    EGShelfLayer,
    MeshCandidate,
    ObjectType,
)
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
    GraphOfBoundingBoxes,
)


def _typed_object(object_type: ObjectType, object_id: str) -> EGObject2D:
    return EGObject2D(
        object_type=object_type,
        scale=Scale(x=0.1, y=0.1, z=0.1),
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        source_id=object_id,
    )


# ---------------------------------------------------------------------------
# free_object_slot - free floor object sampling query
# ---------------------------------------------------------------------------


def test_free_object_slot_leaves_the_whole_pose_free() -> None:
    """
    Free floor objects always sit upright without tilting; ``Pose2D`` has no roll/pitch
    dimensions to pin in the first place (unlike the old ``EGRotation``), so x, y, and
    yaw are all left underspecified for the RSPN to sample.
    """
    pose = free_object_slot().kwargs["pose"]

    assert pose.kwargs["x"] is ...
    assert pose.kwargs["y"] is ...
    assert pose.kwargs["yaw"] is ...


# ---------------------------------------------------------------------------
# build_layer_query - conditioning on fixed objects during RSPN sampling
# ---------------------------------------------------------------------------


def test_build_layer_query_frees_resampled_scale_and_pose() -> None:
    """
    build_layer_query must condition only the fixed objects' scale and pose, leaving a
    free slot's scale, position, and orientation all free to be redrawn.

    Conditioning a resampled slot on its own scale pins the query to the
    single training example that combination of evidence (its own scale plus
    every fixed neighbour's exact pose) came from, collapsing the RSPN's
    posterior for that slot's position back to its original, still-colliding
    value -- observed as a repair pass that redraws the exact same pose every
    time and so can never actually resolve a collision. Regression test for
    that collapse.
    """
    query = build_layer_query(
        ObjectType.BOTTLE,
        [_typed_object(ObjectType.BOOK, "fixed")],
        1,
    )
    params = UnderspecifiedParameters(query)
    conditioned_names = {
        variable.name
        for variable in params.conditioning_assignments_from_literal_values
    }
    conditioned_positions = [name for name in conditioned_names if "pose.x" in name]
    conditioned_scales = [
        name for name in conditioned_names if "objects[" in name and "scale.x" in name
    ]
    # Only the one fixed object's pose and scale are conditioned; the
    # free slot's are left entirely free.
    assert len(conditioned_positions) == 1
    assert len(conditioned_scales) == 1


# ---------------------------------------------------------------------------
# End-to-end sampling - drawing a shelf and spawning it
# ---------------------------------------------------------------------------


_LOCAL_BOOK_SCENE_DIR = (
    Path.home() / "Documents" / "sage-10k-scenes" / "20251230_060038_layout_fd6894a7"
)
"""
A locally cached sage10k scene holding a real book mesh (``source_id`` ``afd57fb1``, raw
type ``"book"``).

Only present on machines that have downloaded the sage10k mesh cache -- the repository
ships no book-shaped textured mesh of its own (only ``chair.ply``, whose loader also
requires the exact texcoord layout sage10k's PLY export uses, so it cannot be swapped
for a synthetic primitive either). The test is skipped rather than failing where this
path does not exist, e.g. in CI.
"""

_LOCAL_BOOK_SOURCE_ID = "afd57fb1"
"""
``source_id`` of the real book mesh within :data:`_LOCAL_BOOK_SCENE_DIR`.
"""


@pytest.mark.skipif(
    not (_LOCAL_BOOK_SCENE_DIR / "objects" / f"{_LOCAL_BOOK_SOURCE_ID}.ply").exists(),
    reason="Local sage10k mesh cache is not available on this machine.",
)
def test_a_sampled_one_layer_one_object_shelf_spawns_its_object_in_a_fresh_world() -> (
    None
):
    """
    :meth:`EGShelf.create_in_world` must actually place a sampled object's body in the
    world it creates, not just report it in the sampled schema -- :meth:`EGShelf.spawn`
    silently drops an object it cannot find a mesh for.
    """
    shelf = EGShelf(
        scale=Scale(x=0.3, y=0.4, z=1.0),
        layers=[
            EGShelfLayer(
                objects=[_typed_object(ObjectType.BOOK, "book_0")],
                theme_dominant_type=ObjectType.BOOK,
            )
        ],
        theme_dominant_type=ObjectType.BOOK,
    )
    model = RelationalProbabilisticCircuit(EGShelf, min_samples_per_leaf=1.0).fit(
        [to_dao(shelf)]
    )
    backend = probabilistic_backend(model)
    sample: EGShelf = next(
        iter(backend.evaluate(build_theme_shelf_query(ObjectType.BOOK, [1])))
    )

    sample.source_ids = [
        MeshCandidate(
            scene_dir=_LOCAL_BOOK_SCENE_DIR,
            source_id=_LOCAL_BOOK_SOURCE_ID,
            object_type=ObjectType.BOOK,
        )
    ]

    sample.spawn()
    world = sample.world

    assert len(sample.layers) == 1
    assert len(sample.layers[0].objects) == 1
    spawned_object = sample.layers[0].objects[0]
    assert spawned_object.annotation in world.bodies

    # A second, fixed book -- not sampled -- asking the fitted circuit where it most
    # likely goes. mode_query builds its where-condition from this layer's own
    # calculate_free_space() (see shelf_placement._free_space_where_condition), the
    # same call the plotted gcs below uses.
    held_book = EGObject2D(
        object_type=ObjectType.BOOK,
        scale=Scale(x=0.1, y=0.1, z=0.1),
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        source_id=_LOCAL_BOOK_SOURCE_ID,
        name="held_book",
    )
    held_book.spawn(world, mesh_path=_LOCAL_BOOK_SCENE_DIR, x=1.0, y=0.0)

    placed_object, layer_name = mode_query(sample, model, held_book)
    goal_layer = layer_named(sample, layer_name)

    assert goal_layer is sample.layers[0]
    assert abs(float(placed_object.pose.x)) <= sample.scale.x / 2
    assert abs(float(placed_object.pose.y)) <= sample.scale.y / 2

    with rclpy_node() as node:
        viz_marker = VizMarkerPublisher(node=node, _world=world)
        viz_marker.with_tf_publisher()
        gcs: GraphOfBoundingBoxes = sample.layers[0].annotation.calculate_free_space()
        gcs.plot_and_show_free_space()
        gcs.plot_and_show_occupied_space()
