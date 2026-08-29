from __future__ import annotations

import dataclasses
import shutil
from importlib.resources import files
from itertools import combinations
from pathlib import Path

import pytest
import trimesh

from experiments.scene_generation_experiments.in_world_resolver import (
    InWorldLayoutResolver,
    ShelfLayerGroup,
    minimal_resample_set,
)
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGShelf,
    EGShelfLayer,
    MeshCandidate,
    ObjectType,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    ShelfLayer,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body


_RESOURCES_PLY = (
    Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
)
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
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
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
        object_type=ObjectType.BOOK,
        scale=Scale(x=0.3, y=0.3, z=0.4),
        pose=Pose2D(x=x, y=y, yaw=0.0),
        source_id="test_object",
        name=object_id,
    )


def _shelf(objects: list[EGObject2D], candidate: MeshCandidate) -> EGShelf:
    """
    A single-layer shelf on a generously sized slab, so a resolved object is
    still comfortably supported after being moved a metre away.
    """
    layer = EGShelfLayer(
        objects=objects,
        theme_dominant_type=ObjectType.BOOK,
    )
    return EGShelf(
        scale=Scale(x=4.0, y=4.0, z=2.0),
        layers=[layer],
        source_ids=[candidate],
        theme_dominant_type=ObjectType.BOOK,
    )


def _object_bodies(layer: EGShelfLayer) -> dict[int, Body]:
    """
    The bodies spawned for *layer*'s objects, keyed by their index in
    :attr:`EGShelfLayer.objects`; objects with no spawned body are omitted.
    """
    return {
        index: obj.annotation
        for index, obj in enumerate(layer.objects)
        if obj.annotation is not None
    }


def _colliding_bodies(shelf: EGShelf) -> bool:
    """
    True if any two spawned object bodies on the first layer collide.
    """
    bodies = list(_object_bodies(shelf.layers[0]).values())
    detector = FCLCollisionDetector(_world=shelf.world)
    matrix = CollisionMatrix(
        collision_checks={
            CollisionCheck(body_a=body_a, body_b=body_b, distance=0.0)
            for body_a, body_b in combinations(bodies, 2)
        }
    )
    return detector.check_collisions(matrix).any()


def _multi_layer_shelf(candidate: MeshCandidate, corpus_height: float) -> EGShelf:
    """
    A four-layer shelf whose layers carry no dimensions of their own, so the
    corpus height alone determines how the layers are spread vertically.
    """
    layers = [
        EGShelfLayer(
            objects=[],
            theme_dominant_type=ObjectType.BOOK,
        )
        for _ in range(4)
    ]
    return EGShelf(
        scale=Scale(x=4.0, y=4.0, z=corpus_height),
        layers=layers,
        source_ids=[candidate],
        theme_dominant_type=ObjectType.BOOK,
    )


# ---------------------------------------------------------------------------
# minimal_resample_set - greedy minimum vertex cover of the collision graph
# ---------------------------------------------------------------------------


def test_minimal_resample_set_picks_one_index_for_a_simple_pair() -> None:
    """
    For a single colliding pair, exactly one of the two indices must be chosen,
    deterministically (the higher one, by the tie-break rule).
    """
    assert minimal_resample_set({(0, 1)}) == {1}


def test_minimal_resample_set_picks_the_shared_index_for_a_star_collision() -> None:
    """
    When one index collides with two others that do not collide with each other,
    discarding just the shared index resolves every collision -- that minimal,
    single-index set must be returned, not a larger valid-but-wasteful cover.
    """
    assert minimal_resample_set({(0, 1), (0, 2)}) == {0}


def test_minimal_resample_set_is_empty_without_collisions() -> None:
    """
    With no colliding pairs, nothing needs resampling.
    """
    assert minimal_resample_set(set()) == set()


def test_shelf_layers_are_spread_across_the_corpus_height(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    A multi-layer shelf must place its layers at distinct, increasing heights
    that span its corpus, so passing the shelf's own height (not a layer's slab
    thickness) keeps the layers from collapsing onto the floor.
    """
    corpus_height = 2.0
    shelf = _multi_layer_shelf(mesh_candidate, corpus_height)
    shelf.spawn()

    layer_heights = [
        layer.annotation.root.global_pose.to_position().to_np()[2]
        for layer in shelf.layers
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
    sampled.scale = Scale(x=5.0, y=5.0, z=5.0)
    shelf = _shelf([sampled], mesh_candidate)

    shelf.spawn()
    body = _object_bodies(shelf.layers[0])[0]

    native_extents = trimesh.load(
        str(mesh_candidate.scene_dir / "objects" / "test_object.ply"), process=False
    ).extents
    spawned_extents = body.collision.combined_mesh.extents

    assert spawned_extents == pytest.approx(native_extents, abs=1e-3)


def _single_layer_shelf_with(candidate: MeshCandidate, object_scale: Scale) -> EGShelf:
    """
    A generous single-layer shelf holding one object, so only the candidate's
    own size decides whether it is placed.
    """
    obj = EGObject2D(
        object_type=ObjectType.BOOK,
        scale=object_scale,
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
        source_id="test_object",
    )
    return EGShelf(
        scale=Scale(x=1.0, y=1.0, z=2.0),
        layers=[
            EGShelfLayer(
                objects=[obj],
                theme_dominant_type=ObjectType.BOOK,
            )
        ],
        theme_dominant_type=ObjectType.BOOK,
        source_ids=[candidate],
    )


def test_object_too_big_for_the_layer_is_dropped(mesh_candidate: MeshCandidate) -> None:
    """
    An object whose only available mesh is taller than the layer's clearance
    must be left out of the spawned shelf, since the resolver moves objects only
    in the plane and could never repair a mesh piercing the shelf above.
    """
    too_tall = dataclasses.replace(mesh_candidate, native_extents=(0.1, 0.1, 2.0))
    shelf = _single_layer_shelf_with(too_tall, Scale(x=0.1, y=0.1, z=0.1))

    shelf.spawn()

    assert _object_bodies(shelf.layers[0]) == {}


def test_object_that_fits_the_layer_is_kept(mesh_candidate: MeshCandidate) -> None:
    """
    An object with a mesh that fits the layer's clearance and footprint must be
    spawned as usual.
    """
    fitting = dataclasses.replace(mesh_candidate, native_extents=(0.1, 0.1, 0.1))
    shelf = _single_layer_shelf_with(fitting, Scale(x=0.1, y=0.1, z=0.1))

    shelf.spawn()

    assert set(_object_bodies(shelf.layers[0])) == {0}


def test_create_in_world_still_returns_a_world(mesh_candidate: MeshCandidate) -> None:
    """
    The spawn refactor must keep :meth:`EGShelf.create_in_world` returning a
    plain :class:`World`, so existing callers stay unaffected.
    """
    shelf = _shelf([_object("book_0", 0.0, 0.0)], mesh_candidate)
    assert isinstance(shelf.create_in_world(), World)


def test_spawn_returns_a_body_per_object_and_a_layer_annotation(
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
    shelf.spawn()

    assert len(shelf.layers) == 1
    assert set(_object_bodies(shelf.layers[0])) == {0, 1}
    assert isinstance(shelf.layers[0].annotation, ShelfLayer)


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
    shelf.spawn()
    body = _object_bodies(shelf.layers[0])[0]

    resting_z = body.parent_connection.origin.to_position().to_np()[2]
    expected = shelf.object_local_pose(
        shelf.layers[0].objects[0], resting_z, shelf.corpus
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
    shelf.spawn()
    corpus = shelf.corpus
    object_body = _object_bodies(shelf.layers[0])[0]

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


def test_spawn_keeps_edge_object_clear_of_the_corpus_walls(
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
        objects=[edge_object],
        theme_dominant_type=ObjectType.BOOK,
    )
    shelf = EGShelf(
        scale=Scale(x=_CHAIR_EXTENTS[0], y=_CHAIR_EXTENTS[1], z=2.0),
        layers=[layer],
        source_ids=[mesh_candidate],
        theme_dominant_type=ObjectType.BOOK,
    )

    shelf.spawn()
    corpus_body = shelf.world.get_semantic_annotations_by_type(Cabinet)[0].root
    detector = FCLCollisionDetector(_world=shelf.world)
    matrix = CollisionMatrix(
        collision_checks={
            CollisionCheck(body_a=corpus_body, body_b=body, distance=0.0)
            for body in _object_bodies(shelf.layers[0]).values()
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
    shelf.spawn()
    spawned_layer = shelf.layers[0]
    group = ShelfLayerGroup(
        bodies=_object_bodies(spawned_layer),
        supporting_body=spawned_layer.annotation.root,
        shelf=shelf,
        layer_index=0,
        corpus=shelf.corpus,
    )

    assert group.unsupported_indices() == set()

    group.bodies[0].parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(50.0, 50.0, 0.11)
    )
    assert group.unsupported_indices() == {0}


def test_resolver_drops_a_really_colliding_object(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    Two objects spawned on top of each other must be resolved by dropping one,
    leaving the layer collision-free under a real-mesh check -- the shrunk resolver no
    longer moves anything, since its own resampling now happens before spawning (see
    :class:`~experiments.scene_generation_experiments.pre_spawn_resolver.
    PreSpawnLayoutResolver`); it only cleans up whatever that pre-spawn approximation
    still let through.
    """
    shelf = _shelf(
        [_object("book_0", 0.0, 0.0), _object("book_1", 0.0, 0.0)], mesh_candidate
    )
    shelf.spawn()

    resolver = InWorldLayoutResolver.for_shelf(shelf)
    spawned = resolver.resolve()

    assert not _colliding_bodies(spawned)
    assert len(_object_bodies(spawned.layers[0])) == 1
    assert resolver.dropped_body_count == 1


def test_resolver_drops_an_object_colliding_with_the_corpus_walls(
    mesh_candidate: MeshCandidate,
) -> None:
    """
    An object sampled close enough to a small shelf's edge to collide with the corpus
    wall must be dropped, since the shrunk resolver only checks collisions between a
    layer's own objects and against the shelf's own corpus and no longer moves
    anything.
    """
    # A layer with room to spare around the object's native footprint, so a
    # centred object would clear the walls but one pushed toward the edge pokes
    # into the corpus wall.
    layer_length = _CHAIR_EXTENTS[0] * 1.4
    layer_width = _CHAIR_EXTENTS[1] * 1.4
    layer = EGShelfLayer(
        objects=[_object("edge_book", _CHAIR_EXTENTS[0] * 0.5, 0.0)],
        theme_dominant_type=ObjectType.BOOK,
    )
    shelf = EGShelf(
        scale=Scale(x=layer_length, y=layer_width, z=2.0),
        layers=[layer],
        source_ids=[mesh_candidate],
        theme_dominant_type=ObjectType.BOOK,
    )
    shelf.spawn()

    resolver = InWorldLayoutResolver.for_shelf(shelf)
    spawned = resolver.resolve()

    assert _object_bodies(spawned.layers[0]) == {}
    assert resolver.dropped_body_count == 1


def test_resolver_leaves_a_collision_free_shelf_untouched(
    mesh_candidate: MeshCandidate,
) -> None:
    shelf = _shelf(
        [_object("book_0", 0.0, 0.0), _object("book_1", 1.5, 0.0)], mesh_candidate
    )
    shelf.spawn()

    resolver = InWorldLayoutResolver.for_shelf(shelf)
    spawned = resolver.resolve()

    assert set(_object_bodies(spawned.layers[0])) == {0, 1}
    assert resolver.dropped_body_count == 0
