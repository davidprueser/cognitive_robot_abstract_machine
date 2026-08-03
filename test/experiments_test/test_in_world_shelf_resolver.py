from __future__ import annotations

import dataclasses
import shutil
from importlib.resources import files
from itertools import combinations
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import trimesh

from experiments.scene_generation_experiments.in_world_resolver import (
    InWorldLayoutResolver,
    ShelfLayerGroup,
)
from krrood.entity_query_language.exceptions import NoSolutionFound
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGPoint2D,
    EGRotation,
    EGScale,
    EGShelf,
    EGShelfLayer,
    MeshCandidate,
    ObjectType,
    SpawnedShelf,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    ShelfLayer,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF


_RESOURCES_PLY = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
_CHAIR_EXTENTS = trimesh.load(str(_RESOURCES_PLY / "chair.ply"), process=False).extents
"""
Native (x, y, z) bounding-box size of the bundled chair mesh. Objects now spawn
at their mesh's real size, so corpus-wall fixtures are sized relative to this.
"""


@pytest.fixture
def mesh_candidate(tmp_path: Path) -> MeshCandidate:
    """
    A mesh candidate backed by the bundled chair PLY, so objects spawn with real
    geometry the FCL detector and supporting-surface checks can act on.
    """
    resources_root = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "test_object.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "test_object_texture.png"
    )
    return MeshCandidate(
        scene_dir=tmp_path, source_id="test_object", object_type=ObjectType.BOOK
    )


def _object(object_id: str, x: float, y: float) -> EGObject2D:
    return EGObject2D(
        id=object_id,
        room_id="room_1",
        place_id="shelf_1",
        object_type=ObjectType.BOOK,
        scale=EGScale(height=0.4, length=0.3, width=0.3),
        position=EGPoint2D(x=x, y=y),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id="test_object",
    )


def _shelf(objects: list[EGObject2D], candidate: MeshCandidate) -> EGShelf:
    """
    A single-layer shelf on a generously sized slab, so a resolved object is
    still comfortably supported after being moved a metre away.
    """
    layer = EGShelfLayer(
        scale=EGScale(height=0.02, length=4.0, width=4.0), objects=objects
    )
    return EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(height=2.0, length=4.0, width=4.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        layers=[layer],
        source_ids=[candidate],
    )


def _colliding_bodies(spawned: SpawnedShelf) -> bool:
    """
    True if any two spawned object bodies on the first layer collide.
    """
    bodies = list(spawned.layers[0].object_bodies.values())
    detector = FCLCollisionDetector(_world=spawned.world)
    matrix = CollisionMatrix(
        collision_checks={
            CollisionCheck(body_a=body_a, body_b=body_b, distance=0.0)
            for body_a, body_b in combinations(bodies, 2)
        }
    )
    return detector.check_collisions(matrix).any()


def _multi_layer_shelf(candidate: MeshCandidate, corpus_height: float) -> EGShelf:
    """
    A four-layer shelf whose layers carry only the fixed slab thickness, so the
    corpus height alone determines how the layers are spread vertically.
    """
    layers = [
        EGShelfLayer(scale=EGScale(height=0.02, length=4.0, width=4.0), objects=[])
        for _ in range(4)
    ]
    return EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(height=corpus_height, length=4.0, width=4.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        layers=layers,
        source_ids=[candidate],
    )


def test_shelf_layers_are_spread_across_the_corpus_height(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    A multi-layer shelf must place its layers at distinct, increasing heights
    that span its corpus, so passing the shelf's own height (not a layer's slab
    thickness) keeps the layers from collapsing onto the floor.
    """
    corpus_height = 2.0
    spawned = _multi_layer_shelf(mesh_candidate, corpus_height).spawn_in_world()

    layer_heights = [
        layer.surface.root.global_pose.to_position().to_np()[2]
        for layer in spawned.layers
    ]

    assert layer_heights == sorted(layer_heights)
    assert len(set(layer_heights)) == len(layer_heights)
    assert max(layer_heights) - min(layer_heights) > corpus_height / 2


def test_object_spawns_at_its_meshs_native_size(mesh_candidate: MeshCandidate) -> None:
    """
    A spawned object must keep its mesh's own real-world size, since sage10k
    meshes already have their scale baked in. Stretching the mesh to an
    RSPN-sampled scale distorts it, so the sampled scale must not override the
    mesh's native extents.
    """
    sampled = _object("book_0", 0.0, 0.0)
    sampled.scale = EGScale(height=5.0, length=5.0, width=5.0)
    shelf = _shelf([sampled], mesh_candidate)

    spawned = shelf.spawn_in_world()
    body = spawned.layers[0].object_bodies[0]

    native_extents = trimesh.load(
        str(mesh_candidate.scene_dir / "objects" / "test_object.ply"), process=False
    ).extents
    spawned_extents = body.collision.combined_mesh.extents

    assert spawned_extents == pytest.approx(native_extents, abs=1e-3)


def _single_layer_shelf_with(
    candidate: MeshCandidate, object_scale: EGScale
) -> EGShelf:
    """
    A generous single-layer shelf holding one object, so only the candidate's
    own size decides whether it is placed.
    """
    obj = EGObject2D(
        id="obj_0",
        room_id="room_1",
        place_id="shelf_1",
        object_type=ObjectType.BOOK,
        scale=object_scale,
        position=EGPoint2D(x=0.0, y=0.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        source_id="test_object",
    )
    return EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(height=2.0, length=1.0, width=1.0),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        layers=[EGShelfLayer(scale=EGScale(height=0.02, length=1.0, width=1.0), objects=[obj])],
        source_ids=[candidate],
    )


def test_object_too_big_for_the_layer_is_dropped(mesh_candidate: MeshCandidate) -> None:
    """
    An object whose only available mesh is taller than the layer's clearance
    must be left out of the spawned shelf, since the resolver moves objects only
    in the plane and could never repair a mesh piercing the shelf above.
    """
    too_tall = dataclasses.replace(
        mesh_candidate, native_extents=(0.1, 0.1, 2.0)
    )
    shelf = _single_layer_shelf_with(too_tall, EGScale(height=0.1, length=0.1, width=0.1))

    spawned = shelf.spawn_in_world()

    assert spawned.layers[0].object_bodies == {}


def test_object_that_fits_the_layer_is_kept(mesh_candidate: MeshCandidate) -> None:
    """
    An object with a mesh that fits the layer's clearance and footprint must be
    spawned as usual.
    """
    fitting = dataclasses.replace(
        mesh_candidate, native_extents=(0.1, 0.1, 0.1)
    )
    shelf = _single_layer_shelf_with(fitting, EGScale(height=0.1, length=0.1, width=0.1))

    spawned = shelf.spawn_in_world()

    assert set(spawned.layers[0].object_bodies) == {0}


def test_create_in_world_still_returns_a_world(mesh_candidate: MeshCandidate) -> None:
    """
    The spawn refactor must keep :meth:`EGShelf.create_in_world` returning a
    plain :class:`World`, so existing callers stay unaffected.
    """
    shelf = _shelf([_object("book_0", 0.0, 0.0)], mesh_candidate)
    assert isinstance(shelf.create_in_world(), World)


def test_spawn_in_world_returns_a_body_per_object_and_a_layer_annotation(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    Spawning must hand back a body for every object and the layer's
    supporting-surface annotation, so the resolver can validate and move objects
    without rebuilding the world.
    """
    shelf = _shelf(
        [_object("book_0", 0.0, 0.0), _object("book_1", 1.0, 0.0)], mesh_candidate
    )
    spawned = shelf.spawn_in_world()

    assert len(spawned.layers) == 1
    assert set(spawned.layers[0].object_bodies) == {0, 1}
    assert isinstance(spawned.layers[0].surface, ShelfLayer)


def test_spawned_body_pose_matches_object_local_pose(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    A freshly spawned object body must sit, in the corpus frame, exactly where
    :meth:`EGShelf.object_local_pose` says it should -- pinning the single pose
    formula that both spawning and later moving rely on, so the two can never
    drift.
    """
    shelf = _shelf([_object("book_0", 0.5, -0.3)], mesh_candidate)
    spawned = shelf.spawn_in_world()
    body = spawned.layers[0].object_bodies[0]

    resting_z = body.parent_connection.origin.to_position().to_np()[2]
    expected = shelf.object_local_pose(
        shelf.layers[0].objects[0], resting_z, spawned.corpus
    )
    assert body.parent_connection.origin.to_np() == pytest.approx(expected.to_np())


def test_spawned_shelf_corpus_is_movable_as_a_unit(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    The shelf corpus must hang off its parent by a movable 6-DoF connection, so
    a room-level resolver can reposition the whole shelf -- corpus, slabs, and
    objects -- in place by setting the corpus origin, and its contents follow.
    """
    shelf = _shelf([_object("book_0", 0.0, 0.0)], mesh_candidate)
    spawned = shelf.spawn_in_world()
    corpus = spawned.corpus
    object_body = spawned.layers[0].object_bodies[0]

    assert isinstance(corpus.parent_connection, Connection6DoF)

    before = object_body.global_pose.to_position().to_np()
    corpus_origin = corpus.parent_connection.origin
    shifted = HomogeneousTransformationMatrix.from_xyz_rpy(
        corpus_origin.to_position().to_np()[0] + 2.0,
        corpus_origin.to_position().to_np()[1],
        corpus_origin.to_position().to_np()[2],
        reference_frame=corpus_origin.reference_frame,
    )
    corpus.parent_connection.origin = shifted

    after = object_body.global_pose.to_position().to_np()
    assert after[0] == pytest.approx(before[0] + 2.0)
    assert after[1] == pytest.approx(before[1])


def test_spawn_in_world_keeps_edge_object_clear_of_the_corpus_walls(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    An object filling a layer edge-to-edge must not collide with the spawned
    :class:`Cabinet` corpus's walls.

    The corpus used to be sized exactly to the layers' footprint and then had
    a wall carved out of that same footprint, so the wall intruded into the
    region objects were trained to occupy. Sizing the layer to the object's own
    native footprint presses it against every wall at once.
    """
    edge_object = _object("edge_book", 0.0, 0.0)
    layer = EGShelfLayer(
        scale=EGScale(
            height=0.02, length=_CHAIR_EXTENTS[0], width=_CHAIR_EXTENTS[1]
        ),
        objects=[edge_object],
    )
    shelf = EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(height=2.0, length=_CHAIR_EXTENTS[0], width=_CHAIR_EXTENTS[1]),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        layers=[layer],
        source_ids=[mesh_candidate],
    )

    spawned = shelf.spawn_in_world()
    corpus_body = spawned.world.get_semantic_annotations_by_type(Cabinet)[0].root
    detector = FCLCollisionDetector(_world=spawned.world)
    matrix = CollisionMatrix(
        collision_checks={
            CollisionCheck(body_a=corpus_body, body_b=body, distance=0.0)
            for body in spawned.layers[0].object_bodies.values()
        }
    )
    assert not detector.check_collisions(matrix).any()


def test_unsupported_indices_flags_object_that_slid_off_the_layer(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    An object moved off the slab must be reported as unsupported -- the
    annotation-based replacement for the old out-of-bounds footprint check.
    """
    from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

    shelf = _shelf([_object("book_0", 0.0, 0.0)], mesh_candidate)
    spawned = shelf.spawn_in_world()
    spawned_layer = spawned.layers[0]
    group = ShelfLayerGroup(
        bodies=spawned_layer.object_bodies,
        supporting_body=spawned_layer.surface.root,
        backend=MagicMock(),
        shelf=shelf,
        layer_index=0,
        corpus=spawned.corpus,
    )

    assert group.unsupported_indices() == set()

    group.bodies[0].parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(50.0, 50.0, 0.11)
    )
    assert group.unsupported_indices() == {0}


def test_resolver_moves_colliding_object_until_layer_is_collision_free(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    Two overlapping objects must be resolved by moving one body to a redrawn,
    separated pose, leaving the layer collision-free under a real-mesh check.
    """
    shelf = _shelf(
        [_object("book_0", 0.0, 0.0), _object("book_1", 0.0, 0.0)], mesh_candidate
    )
    separated_layer = EGShelfLayer(
        scale=shelf.layers[0].scale,
        objects=[_object("fixed", 0.0, 0.0), _object("moved", 0.0, 1.5)],
    )

    with patch(
        "experiments.scene_generation_experiments.in_world_resolver.probabilistic_backend"
    ) as backend_factory:
        backend_factory.return_value.evaluate.return_value = [separated_layer]
        resolver = InWorldLayoutResolver.for_shelf(shelf, rspn=MagicMock())
        spawned = resolver.resolve()

    assert shelf.layers[0].objects[1].position == EGPoint2D(x=0.0, y=1.5)
    assert not _colliding_bodies(spawned)


def test_resolver_moves_object_colliding_with_the_corpus_walls(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    An object sampled close enough to a small shelf's edge to collide with the
    corpus wall must be resolved by the repair loop.

    The loop previously only checked collisions between an layer's own
    objects, never against the shelf's own corpus -- so an object placed
    inside a wall was never flagged and stayed there.
    """
    # A layer with room to spare around the object's native footprint, so a
    # centred object clears the walls but one pushed toward the edge pokes into
    # the corpus wall.
    layer_length = _CHAIR_EXTENTS[0] * 1.4
    layer_width = _CHAIR_EXTENTS[1] * 1.4
    layer = EGShelfLayer(
        scale=EGScale(height=0.02, length=layer_length, width=layer_width),
        objects=[_object("edge_book", _CHAIR_EXTENTS[0] * 0.5, 0.0)],
    )
    shelf = EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(height=2.0, length=layer_length, width=layer_width),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        layers=[layer],
        source_ids=[mesh_candidate],
    )
    centered_layer = EGShelfLayer(
        scale=layer.scale, objects=[_object("moved", 0.0, 0.0)]
    )

    with patch(
        "experiments.scene_generation_experiments.in_world_resolver.probabilistic_backend"
    ) as backend_factory:
        backend_factory.return_value.evaluate.return_value = [centered_layer]
        resolver = InWorldLayoutResolver.for_shelf(shelf, rspn=MagicMock())
        spawned = resolver.resolve()

    corpus_body = spawned.world.get_semantic_annotations_by_type(Cabinet)[0].root
    detector = FCLCollisionDetector(_world=spawned.world)
    matrix = CollisionMatrix(
        collision_checks={
            CollisionCheck(body_a=corpus_body, body_b=body, distance=0.0)
            for body in spawned.layers[0].object_bodies.values()
        }
    )
    assert not detector.check_collisions(matrix).any()


def test_resolver_falls_back_to_relaxed_query_when_neighbour_evidence_has_no_solution(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    When the neighbour-conditioned resample query has no support in the
    fitted circuit -- a real failure mode once an object's pose has drifted
    through several repair passes -- the resolver must retry without the
    fixed neighbour's evidence instead of letting NoSolutionFound abort the
    whole repair.
    """
    shelf = _shelf(
        [_object("book_0", 0.0, 0.0), _object("book_1", 0.0, 0.0)], mesh_candidate
    )
    relaxed_layer = EGShelfLayer(
        scale=shelf.layers[0].scale, objects=[_object("moved", 0.0, 1.5)]
    )

    with patch(
        "experiments.scene_generation_experiments.in_world_resolver.probabilistic_backend"
    ) as backend_factory:
        backend_factory.return_value.evaluate.side_effect = [
            NoSolutionFound(expression=MagicMock(), found_number=0),
            [relaxed_layer],
        ]
        resolver = InWorldLayoutResolver.for_shelf(shelf, rspn=MagicMock())
        spawned = resolver.resolve()

    assert backend_factory.return_value.evaluate.call_count == 2
    assert shelf.layers[0].objects[1].position == EGPoint2D(x=0.0, y=1.5)
    assert not _colliding_bodies(spawned)


def test_resolver_drops_objects_it_cannot_separate(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    When resampling never separates the objects, the resolver must give up
    moving them and drop the offenders, returning a collision-free layout rather
    than spinning forever or failing the whole sample.
    """
    shelf = _shelf(
        [_object("book_0", 0.0, 0.0), _object("book_1", 0.0, 0.0)], mesh_candidate
    )
    still_overlapping = EGShelfLayer(
        scale=shelf.layers[0].scale,
        objects=[_object("fixed", 0.0, 0.0), _object("moved", 0.0, 0.0)],
    )

    with patch(
        "experiments.scene_generation_experiments.in_world_resolver.probabilistic_backend"
    ) as backend_factory:
        backend_factory.return_value.evaluate.return_value = [still_overlapping]
        resolver = InWorldLayoutResolver.for_shelf(shelf, rspn=MagicMock(), max_passes=3)
        spawned = resolver.resolve()

    assert not _colliding_bodies(spawned)
    assert len(spawned.layers[0].object_bodies) < 2
