from __future__ import annotations

import math
import shutil
from importlib.resources import files
from pathlib import Path

import pytest

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGPoint2D,
    EGRotation,
    EGShelf,
    EGShelfLayer,
    EGScale,
    MeshCandidate,
    ObjectType,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def chair_mesh_directory(tmp_path: Path) -> Path:
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_dir / "chair_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png", objects_dir / "chair_src_texture.png"
    )
    return tmp_path


@pytest.fixture
def close_and_oversized_book_candidates(
    tmp_path: Path,
) -> tuple[MeshCandidate, MeshCandidate]:
    """
    Two mesh candidates of the same ``ObjectType``, sharing the same underlying asset
    but claiming very different real-world sizes -- one close to the book object
    :func:`_make_layer` samples, one bookcase-sized.

    The geometry itself is irrelevant here; only the claimed ``native_extents`` matter
    for the size scoring under test.
    """
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
    objects_dir = tmp_path / "objects"
    objects_dir.mkdir()
    for source_id in ("close_match", "oversized"):
        shutil.copy(resources_root / "chair.ply", objects_dir / f"{source_id}.ply")
        shutil.copy(
            resources_root / "chair_texture.png",
            objects_dir / f"{source_id}_texture.png",
        )
    return (
        MeshCandidate(
            tmp_path, "close_match", ObjectType.BOOK, native_extents=(0.1, 0.05, 0.2)
        ),
        MeshCandidate(
            tmp_path, "oversized", ObjectType.BOOK, native_extents=(0.6, 0.25, 1.5)
        ),
    )


def _make_layer(
    relative_height: float = 0.0, width: float = 0.8, length: float = 0.4
) -> EGShelfLayer:
    return EGShelfLayer(
        scale=EGScale(width=width, length=length, height=0.02),
        objects=[
            EGObject2D(
                id="book_1",
                room_id="room_1",
                place_id="shelf_1",
                object_type=ObjectType.BOOK,
                scale=EGScale(width=0.1, length=0.05, height=0.2),
                position=EGPoint2D(x=0.0, y=0.0),
                orientation=EGRotation(x=0.0, y=0.0, z=0.0),
                source_id="chair_src",
                theme_dominant_type=ObjectType.BOOK,
            )
        ],
        theme_dominant_type=ObjectType.BOOK,
        relative_height=relative_height,
    )


def _make_shelf(
    relative_heights: tuple[float, ...] = (0.0,),
    scale: EGScale | None = None,
    layer_scales: tuple[tuple[float, float], ...] | None = None,
) -> EGShelf:
    """
    A shelf whose layers may deliberately disagree with it in footprint, which is what
    an independently sampled shelf looks like before the footprint is pinned.
    """
    footprints = layer_scales or tuple((0.8, 0.4) for _ in relative_heights)
    return EGShelf(
        scale=scale or EGScale(height=2.0, length=0.4, width=0.8),
        layers=[
            _make_layer(height, width, length)
            for height, (width, length) in zip(relative_heights, footprints)
        ],
        theme_dominant_type=ObjectType.BOOK,
        source_ids=[],
    )


def _slab_footprints(shelf: EGShelf) -> set[tuple[float, float]]:
    """
    Every spawned slab's x/y extents, rounded so float noise does not split them.
    """
    return {
        (
            round(float(spawned_layer.surface.root.collision.shapes[0].scale.x), 6),
            round(float(spawned_layer.surface.root.collision.shapes[0].scale.y), 6),
        )
        for spawned_layer in shelf.spawn_in_world().layers
    }


def _slab_heights(shelf: EGShelf) -> list[float]:
    """
    Height of every spawned slab above the shelf's base, lowest first.

    Slabs are reparented onto the corpus so the shelf moves as one unit, which puts
    their origins in the corpus frame, centred half a shelf up.
    """
    return sorted(
        float(
            spawned_layer.surface.root.parent_connection.origin.to_position()
            .to_np()[2]
            .item()
        )
        + shelf.scale.height / 2
        for spawned_layer in shelf.spawn_in_world().layers
    )


def test_shelf_mounts_at_the_origin_of_the_parent_it_is_given(
    chair_mesh_directory: Path,
) -> None:
    """
    A shelf defines the frame its contents are expressed in and sits at that frame's
    origin, so a caller positions it by choosing the parent rather than by a pose on the
    shelf.

    The corpus is still built in the content frame, so its yaw is
    :attr:`EGShelf.CONTENT_FRAME_YAW_OFFSET_DEGREES`, which extraction has to invert.
    """
    shelf = _make_shelf()
    shelf.source_ids = [
        MeshCandidate(chair_mesh_directory, "chair_src", ObjectType.BOOK)
    ]

    world = World()
    parent = Body(name=PrefixedName(name="room_parent"))
    with world.modify_world():
        world.add_body(parent)

    shelf.create_in_world(world, parent=parent)

    [corpus_body] = [body for body in world.bodies if body.name.name == "shelf_corpus"]
    assert corpus_body.parent_connection.parent is parent

    translation = corpus_body.parent_connection.origin.to_position().to_np()
    assert translation[0] == pytest.approx(0.0, abs=1e-6)
    assert translation[1] == pytest.approx(0.0, abs=1e-6)

    expected_yaw = math.radians(EGShelf.CONTENT_FRAME_YAW_OFFSET_DEGREES)
    yaw = corpus_body.parent_connection.origin.to_rotation_matrix().to_rpy()[2]
    assert float(yaw.to_np().item()) == pytest.approx(expected_yaw, abs=1e-6)


def test_theme_dominant_type_survives_a_json_round_trip() -> None:
    """
    The theme is what a sampled shelf is conditioned on, so a stored shelf that lost it
    would come back as a shelf of no particular theme.
    """
    shelf = _make_shelf()

    restored = EGShelf.from_json(shelf.to_json())

    assert restored.theme_dominant_type is ObjectType.BOOK
    assert [layer.theme_dominant_type for layer in restored.layers] == [
        ObjectType.BOOK
    ]


def test_every_slab_spawns_at_the_shelfs_own_footprint() -> None:
    """
    Layers are drawn independently, so their footprints can disagree with each other and
    with the shelf.

    Spawning each at its own size leaves smaller slabs floating clear of the corpus
    walls, so the shelf's footprint is the single source of truth.
    """
    shelf = _make_shelf(
        relative_heights=(0.2, 0.5, 0.8),
        layer_scales=((0.8, 0.4), (0.5, 0.2), (0.7, 0.3)),
    )

    footprints = _slab_footprints(shelf)

    assert footprints == {(shelf.scale.length, shelf.scale.width)}


def test_the_corpus_interior_matches_the_shelfs_learned_dimensions() -> None:
    """
    The corpus used to be sized from the layers, so the shelf's own learned width and
    depth had no effect and every type spawned the same box.

    They are what distinguishes a narrow bookcase from a wide cabinet.
    """
    narrow = _make_shelf(scale=EGScale(height=1.5, length=0.26, width=0.62))
    wide = _make_shelf(scale=EGScale(height=1.5, length=0.40, width=1.43))

    assert narrow.corpus_footprint.width == pytest.approx(
        0.62 + 2 * EGShelf._CORPUS_WALL_THICKNESS, abs=1e-6
    )
    assert wide.corpus_footprint.width == pytest.approx(
        1.43 + 2 * EGShelf._CORPUS_WALL_THICKNESS, abs=1e-6
    )
    assert narrow.corpus_footprint.width < wide.corpus_footprint.width


def test_slabs_are_evenly_spaced_whatever_heights_were_drawn() -> None:
    """
    ``relative_height`` records where objects were *found*, not where slabs are:

    an empty shelf level leaves no trace, so a measured gap is the distance to the next
    occupied level. Real shelves are evenly spaced, so the drawn heights must not become
    slab positions.
    """
    shelf = _make_shelf(relative_heights=(0.05, 0.06, 0.9))

    heights = _slab_heights(shelf)

    gaps = [second - first for first, second in zip(heights, heights[1:])]
    assert gaps == pytest.approx([gaps[0]] * len(gaps), abs=1e-6)


def test_each_slab_is_placed_at_its_own_layers_height_rank() -> None:
    """
    ``_layer_heights`` computes the evenly-spaced grid by walking ``layers`` sorted by
    ``relative_height``, but a shelf's ``layers`` are drawn from an exchangeable RSPN
    template and come back in no particular order.

    A slab must land at the height rank of the layer it was built for, not at whichever
    grid slot the sorted pass happened to produce it in.
    """
    shelf = _make_shelf(relative_heights=(0.9, 0.1, 0.5))

    spawned_heights = [
        float(
            spawned_layer.surface.root.parent_connection.origin.to_position()
            .to_np()[2]
            .item()
        )
        + shelf.scale.height / 2
        for spawned_layer in shelf.spawn_in_world().layers
    ]

    ranks = sorted(range(len(spawned_heights)), key=lambda index: spawned_heights[index])
    expected_ranks = sorted(
        range(len(shelf.layers)),
        key=lambda index: shelf.layers[index].relative_height,
    )
    assert ranks == expected_ranks


def test_every_slab_gap_leaves_room_for_a_typical_object() -> None:
    """
    Layers drawn close together used to be pushed apart by 3 cm, which is far less than
    anything stands on a shelf, so the layer spawned empty.

    The median object in the dataset is 0.077 m tall.
    """
    shelf = _make_shelf(relative_heights=(0.5, 0.5, 0.5))

    heights = _slab_heights(shelf)

    gaps = [second - first for first, second in zip(heights, heights[1:])]
    assert all(gap > 0.077 for gap in gaps)


def test_objects_recorded_on_top_of_low_furniture_spawn_above_it() -> None:
    """
    A quarter of the recorded layers describe things standing *on* a piece of furniture,
    not on a shelf inside it.

    On a low cabinet that is where they belong, and spawning them inside crushes them
    against the corpus ceiling.
    """
    shelf = _make_shelf(
        relative_heights=(0.3, 1.0),
        scale=EGScale(height=1.0, length=0.4, width=0.8),
    )

    heights = _slab_heights(shelf)

    assert heights[-1] == pytest.approx(shelf.scale.height, abs=1e-6)


def test_objects_recorded_on_top_of_tall_furniture_spawn_inside_it() -> None:
    """
    Nobody leaves things on top of a shelf they cannot reach, so above the reach
    threshold the recorded top layer is an ordinary level.

    Its objects are kept rather than discarded -- they were observed, only their height
    is implausible.
    """
    shelf = _make_shelf(
        relative_heights=(0.3, 1.0),
        scale=EGScale(height=2.4, length=0.4, width=0.8),
    )

    heights = _slab_heights(shelf)

    assert all(height < shelf.scale.height for height in heights)
    assert len(heights) == 2


def test_an_object_on_the_shelfs_top_is_not_rejected_for_lack_of_headroom(
    chair_mesh_directory: Path,
) -> None:
    """
    Nothing stands above the shelf's top, so an object placed there has open air over
    it.

    Measuring its headroom against the corpus ceiling -- which lies *below* the top
    surface -- makes every such object appear too tall and drops the whole layer.
    """
    shelf = _make_shelf(
        relative_heights=(1.0,), scale=EGScale(height=1.0, length=0.4, width=0.8)
    )
    shelf.source_ids = [
        MeshCandidate(
            chair_mesh_directory,
            "chair_src",
            ObjectType.BOOK,
            native_extents=(0.1, 0.05, 0.2),
        )
    ]

    spawned = shelf.spawn_in_world()

    assert sum(len(layer.object_bodies) for layer in spawned.layers) == 1


def test_object_mesh_is_matched_to_its_sampled_size_not_just_its_type(
    close_and_oversized_book_candidates: tuple[MeshCandidate, MeshCandidate],
) -> None:
    """
    A mesh candidate's real size must be weighed against the size the circuit actually
    sampled for that object, not only its ``ObjectType`` -- otherwise a bookcase-sized
    mesh tagged the same type as a 0.2 m book is just as eligible as one the right size,
    purely because both fit the layer's footprint (and, on the shelf's top, its
    unbounded headroom).
    """
    close_match, oversized = close_and_oversized_book_candidates
    shelf = _make_shelf(
        relative_heights=(1.0,), scale=EGScale(height=1.0, length=0.4, width=0.8)
    )
    shelf.source_ids = [oversized, close_match]

    spawned = shelf.spawn_in_world()

    assert sum(len(layer.object_bodies) for layer in spawned.layers) == 1
    assert shelf.layers[0].objects[0].source_id == "close_match"


def test_an_object_with_no_mesh_spawns_a_placeholder_when_asked() -> None:
    """
    Objects whose type has no cached mesh are skipped in silence, so a shelf can render
    nearly empty with nothing to say why.

    Standing a plain box in their place makes the gap between what was sampled and what
    could be shown visible while the mesh library is incomplete.
    """
    shelf = _make_shelf(relative_heights=(0.3,))
    shelf.source_ids = []

    spawned = shelf.spawn_in_world(placeholders_for_missing_meshes=True)

    assert sum(len(layer.object_bodies) for layer in spawned.layers) == 1
    assert spawned.placeholder_count == 1


def test_objects_with_no_mesh_are_dropped_by_default() -> None:
    """
    Placeholders are a diagnostic, so a generation run that is not being inspected must
    not silently gain boxes that stand for nothing.
    """
    shelf = _make_shelf(relative_heights=(0.3,))
    shelf.source_ids = []

    spawned = shelf.spawn_in_world()

    assert sum(len(layer.object_bodies) for layer in spawned.layers) == 0
    assert spawned.placeholder_count == 0


def test_only_one_layer_can_occupy_the_shelfs_top() -> None:
    """
    Layers are drawn independently, so several can come back recorded at the shelf's
    top.

    A shelf has one top, and placing them all there stacks slabs at the same height with
    no room between them for anything to stand.
    """
    shelf = _make_shelf(
        relative_heights=(0.3, 1.0, 1.0),
        scale=EGScale(height=1.4, length=0.4, width=0.8),
    )

    heights = _slab_heights(shelf)

    gaps = [second - first for first, second in zip(heights, heights[1:])]
    assert all(gap > 0.077 for gap in gaps)


def test_a_placeholder_can_be_repositioned_like_a_real_object() -> None:
    """
    Placeholders stand in the object list, so collision repair moves them exactly as it
    moves real objects.

    Attached rigidly they cannot be moved at all, and the first repair pass fails on the
    whole shelf rather than on the stand-in.
    """
    shelf = _make_shelf(relative_heights=(0.3,))
    shelf.source_ids = []

    spawned = shelf.spawn_in_world(placeholders_for_missing_meshes=True)

    [placeholder] = spawned.layers[0].object_bodies.values()
    placeholder.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            0.1, 0.0, 0.0, reference_frame=placeholder.parent_connection.parent
        )
    )
