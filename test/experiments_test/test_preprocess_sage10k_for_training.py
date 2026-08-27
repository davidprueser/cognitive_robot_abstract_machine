from __future__ import annotations

import dataclasses
import math
import shutil
from importlib.resources import files
from pathlib import Path

import pytest
import trimesh
from sqlalchemy.orm import Session

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from experiments.orm.ormatic_interface import (
    Sage10kObjectDAO,
    Sage10kPhysicallyBasedRenderingDAO,
    Sage10kPositionDAO,
    Sage10kRotationDAO,
    Sage10kSizeDAO,
)
from experiments.sage_10k.preprocess_sage10k_for_training import (
    MeshMeasurements,
    ShelfContents,
    eg_object_from_sage10k_object,
    object_type_affinities,
    VerticalExtent,
    object_type_height_profiles,
    shelves_with_layers,
)
from semantic_digital_twin.scene_generation.shelf_membership_classifier import (
    ShelfMembershipClassifier,
)
from semantic_digital_twin.scene_generation.object_type_classifier import (
    ObjectTypeClassifier,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject,
    EGObject2D,
    EGPoint2D,
    EGPosition,
    EGRotation,
    EGShelf,
    EGShelfLayer,
    MeshCandidate,
    ObjectType,
)
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body

_SHELF_ID = "room_1_shelf_1"


def _empty_world() -> tuple[World, Body]:
    """
    A fresh world with a single root body, for spawning a shelf into.
    """
    world = World()
    root = Body(name=PrefixedName(name="map"))
    with world.modify_world():
        world.add_body(root)
    return world, root


def _move_shelf_to(spawned: EGShelf, x: float, y: float, yaw_degrees: float) -> None:
    """
    Put a spawned shelf where it stood in the scene it was extracted from.

    A shelf spawns at its parent's origin, so it is placed by moving its corpus, which
    is the movable branch the whole shelf hangs off. The corpus is built in the content
    frame, so its yaw carries :attr:`EGShelf.CONTENT_FRAME_YAW_OFFSET_DEGREES` on top of
    the shelf's own.
    """
    origin = spawned.corpus.parent_connection.origin
    spawned.corpus.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=x,
            y=y,
            z=float(origin.to_position().to_np()[2].item()),
            yaw=math.radians(yaw_degrees + EGShelf.CONTENT_FRAME_YAW_OFFSET_DEGREES),
            reference_frame=origin.reference_frame,
        )
    )


def _eg_object(
    object_id: str,
    place_id: str,
    object_type: ObjectType,
    x: float,
    y: float,
    z: float = 0.5,
    yaw: float = 0.0,
    width: float = 0.1,
    length: float = 0.1,
    height: float = 0.2,
    source_id: str | None = None,
) -> EGObject:
    return EGObject(
        id=object_id,
        room_id="room_1",
        place_id=place_id,
        object_type=object_type,
        scale=Scale(x=length, y=width, z=height),
        position=EGPosition(x=x, y=y, z=z),
        orientation=EGRotation(x=0.0, y=0.0, z=yaw),
        source_id=source_id or f"{object_id}_src",
    )


def _shelf(
    width: float = 2.0, length: float = 2.0, yaw: float = 0.0, height: float = 2.0
) -> EGObject:
    return _eg_object(
        _SHELF_ID,
        place_id="floor",
        object_type=ObjectType.SHELF,
        x=0.0,
        y=0.0,
        z=1.0,
        yaw=yaw,
        width=width,
        length=length,
        height=height,
    )


def _object_2d(
    object_type: ObjectType, object_id: str, x: float = 0.0, y: float = 0.0
) -> EGObject2D:
    return EGObject2D(
        object_type=object_type,
        scale=Scale(x=0.1, y=0.1, z=0.1),
        pose=Pose2D(x=x, y=y, yaw=0.0),
        source_id=object_id,
        name=object_id,
    )


_SHELF_SOURCE_ID = f"{_SHELF_ID}_src"

_SHELF_EXTENT = VerticalExtent(bottom=-1.0, top=1.0)
"""
Reach of the shelf mesh built by :func:`_shelf`, whose origin sits at z=1.0, so its base
lands at 0.0 and its top at 2.0.
"""


def _layers_by_shelf(objects: list[EGObject], **kwargs) -> list[list[EGShelfLayer]]:
    """
    The extracted shelves' layers, one list per shelf.
    """
    return [
        shelf.layers
        for shelf in shelves_with_layers(
            objects,
            {_SHELF_SOURCE_ID: _SHELF_EXTENT},
            {_SHELF_ID},
            MeshMeasurements(source_id_to_path={}),
            **kwargs,
        )
    ]


def _cache_off_center_mesh(tmp_path: Path, source_id: str) -> Path:
    """
    Write a box mesh, cached under *source_id*, whose local origin sits at a
    corner rather than at its bounding-box centre: it spans x in [0, 0.4] and
    y in [-0.1, 0.1], so its true centre is at local (0.2, 0.0).
    """
    objects_directory = tmp_path / "scene_1" / "objects"
    objects_directory.mkdir(parents=True)
    box = trimesh.creation.box(extents=[0.4, 0.2, 0.2])
    box.apply_translation([0.2, 0.0, 0.1])
    box.export(str(objects_directory / f"{source_id}.ply"))
    return objects_directory.parent


# ---------------------------------------------------------------------------
# MeshMeasurements -- correcting a recorded position to the mesh's centre
# ---------------------------------------------------------------------------


def test_position_is_corrected_to_the_meshs_bounding_box_center(
    tmp_path: Path,
) -> None:
    """
    A sage10k object's recorded position is its mesh's local origin, which is not
    guaranteed to be that mesh's bounding-box centre, so the correction has to shift it
    there.
    """
    scene_directory = _cache_off_center_mesh(tmp_path, "book_src")
    measurements = MeshMeasurements(source_id_to_path={"book_src": scene_directory})

    corrected = measurements.corrected_position(
        source_id="book_src", position=EGPoint2D(x=1.0, y=2.0), yaw_degrees=0.0
    )

    assert corrected.is_mesh_corrected
    assert corrected.position.x == pytest.approx(1.2)
    assert corrected.position.y == pytest.approx(2.0)


def test_position_correction_is_rotated_by_the_objects_own_yaw(
    tmp_path: Path,
) -> None:
    """
    The mesh-local offset has to be rotated into world axes by the object's own yaw, or
    a rotated object is corrected along the wrong axis.
    """
    scene_directory = _cache_off_center_mesh(tmp_path, "book_src")
    measurements = MeshMeasurements(source_id_to_path={"book_src": scene_directory})

    corrected = measurements.corrected_position(
        source_id="book_src", position=EGPoint2D(x=0.0, y=0.0), yaw_degrees=90.0
    )

    assert corrected.position.x == pytest.approx(0.0, abs=1e-9)
    assert corrected.position.y == pytest.approx(0.2)


def test_position_falls_back_to_the_recorded_one_without_a_cached_mesh() -> None:
    """
    An object whose mesh is not cached locally keeps its recorded position and is
    flagged as uncorrected, so the gap stays visible instead of silently mixing
    corrected and uncorrected data.
    """
    measurements = MeshMeasurements(source_id_to_path={})

    corrected = measurements.corrected_position(
        source_id="missing_src", position=EGPoint2D(x=0.3, y=0.1), yaw_degrees=0.0
    )

    assert not corrected.is_mesh_corrected
    assert corrected.position.x == pytest.approx(0.3)
    assert corrected.position.y == pytest.approx(0.1)


def test_mesh_is_measured_once_per_source_id(tmp_path: Path) -> None:
    """
    Many objects share one mesh asset, so measuring a mesh again for every object that
    uses it would dominate the pipeline's runtime.
    """
    scene_directory = _cache_off_center_mesh(tmp_path, "book_src")
    measurements = MeshMeasurements(source_id_to_path={"book_src": scene_directory})

    for _ in range(3):
        measurements.corrected_position(
            source_id="book_src", position=EGPoint2D(x=0.0, y=0.0), yaw_degrees=0.0
        )

    assert measurements.measured_mesh_count == 1


def test_a_measured_position_is_an_ordinary_float(tmp_path: Path) -> None:
    """
    A mesh is measured with numpy, whose scalars pass for floats everywhere except at the
    database driver: PostgreSQL is handed their repr and rejects the statement, while
    SQLite accepts them, so nothing short of an explicit check catches the leak.
    """
    scene_directory = _cache_off_center_mesh(tmp_path, "book_src")
    measurements = MeshMeasurements(source_id_to_path={"book_src": scene_directory})

    corrected = measurements.corrected_position(
        source_id="book_src", position=EGPoint2D(x=1.0, y=2.0), yaw_degrees=0.0
    )

    assert type(corrected.position.x) is float
    assert type(corrected.position.y) is float


# ---------------------------------------------------------------------------
# eg_object_from_sage10k_object -- unified types and carried-through text
# ---------------------------------------------------------------------------


def _sage10k_object(
    raw_type: str = "book2",
    description: str = "A worn hardcover novel",
    place_guidance: str = "on the middle shelf",
    x: float = 0.0,
    y: float = 0.0,
    object_id: str = "book_1",
) -> Sage10kObjectDAO:
    return Sage10kObjectDAO(
        id=object_id,
        room_id="room_1",
        type=raw_type,
        description=description,
        source="generation",
        source_id="book_src",
        place_id=_SHELF_ID,
        place_guidance=place_guidance,
        mass=0.4,
        position=Sage10kPositionDAO(x=x, y=y, z=0.5),
        rotation=Sage10kRotationDAO(x=0.0, y=0.0, z=0.0),
        dimensions=Sage10kSizeDAO(height=0.2, length=0.1, width=0.05),
        pbr_parameters=Sage10kPhysicallyBasedRenderingDAO(metallic=0.0, roughness=0.5),
    )


def test_conversion_maps_the_raw_type_onto_a_unified_object_type() -> None:
    converted = eg_object_from_sage10k_object(
        _sage10k_object(raw_type="bookshelf1"),
        ObjectTypeClassifier(),
        MeshMeasurements(source_id_to_path={}),
    )

    assert converted.object_type == ObjectType.SHELF


def test_conversion_carries_the_datasets_free_text_through() -> None:
    """
    The dataset's ``description`` and ``place_guidance`` are the only natural language
    available for placement reasoning, and were previously dropped on conversion.
    """
    converted = eg_object_from_sage10k_object(
        _sage10k_object(),
        ObjectTypeClassifier(),
        MeshMeasurements(source_id_to_path={}),
    )

    assert converted.description == "A worn hardcover novel"
    assert converted.place_guidance == "on the middle shelf"


def test_conversion_corrects_the_position_and_records_that_it_did(
    tmp_path: Path,
) -> None:
    scene_directory = _cache_off_center_mesh(tmp_path, "book_src")

    converted = eg_object_from_sage10k_object(
        _sage10k_object(),
        ObjectTypeClassifier(),
        MeshMeasurements(source_id_to_path={"book_src": scene_directory}),
    )

    assert converted.position_is_mesh_corrected
    assert converted.position.x == pytest.approx(0.2)
    assert converted.position.z == pytest.approx(0.5)


def test_conversion_flags_an_object_whose_mesh_was_unavailable() -> None:
    converted = eg_object_from_sage10k_object(
        _sage10k_object(x=0.3),
        ObjectTypeClassifier(),
        MeshMeasurements(source_id_to_path={}),
    )

    assert not converted.position_is_mesh_corrected
    assert converted.position.x == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# shelves_with_layers -- grouping corrected objects into ordered layers
# ---------------------------------------------------------------------------


def _rotated_shelf_and_book(local_x: float, local_y: float) -> list[EGObject]:
    """
    A shelf rotated 45 degrees in the room (wide 1.0, shallow 0.4), with one book placed
    at ``(local_x, local_y)`` in the *shelf's own* frame.
    """
    yaw_degrees = 45.0
    theta = math.radians(yaw_degrees)
    shelf = _shelf(width=1.0, length=0.4, yaw=yaw_degrees, height=0.02)
    book = _eg_object(
        "book_1",
        place_id=_SHELF_ID,
        object_type=ObjectType.BOOK,
        x=local_x * math.cos(theta) - local_y * math.sin(theta),
        y=local_x * math.sin(theta) + local_y * math.cos(theta),
    )
    return [shelf, book]


def test_within_bounds_filter_accounts_for_shelf_rotation() -> None:
    """
    A book at the legitimate edge of a rotated shelf's wide axis, centred on its shallow
    axis, must be kept -- comparing the raw world-frame offset against the shelf's width
    and length tests the wrong axes.
    """
    layers_by_shelf = _layers_by_shelf(_rotated_shelf_and_book(0.45, 0.0))

    assert len(layers_by_shelf) == 1
    source_ids = {
        obj.source_id for layer in layers_by_shelf[0] for obj in layer.objects
    }
    assert source_ids == {"book_1_src"}


def test_within_bounds_filter_excludes_object_outside_rotated_footprint() -> None:
    assert _layers_by_shelf(_rotated_shelf_and_book(0.0, 0.35)) == []


def test_layers_are_ordered_from_the_bottom_up() -> None:
    """
    Layer order has to follow height, since a caller reasoning about where on a shelf
    something belongs reads meaning into a layer's position in the list.

    Grouping by cluster label alone leaves the order an accident of which object
    happened to be encountered first.
    """
    objects = [
        _shelf(),
        _eg_object("top", _SHELF_ID, ObjectType.BOOK, x=0.0, y=0.0, z=1.5),
        _eg_object("bottom", _SHELF_ID, ObjectType.BOOK, x=0.1, y=0.0, z=0.2),
        _eg_object("middle", _SHELF_ID, ObjectType.BOOK, x=0.2, y=0.0, z=0.9),
    ]

    [layers] = _layers_by_shelf(objects)

    assert [layer.objects[0].source_id for layer in layers] == [
        "bottom_src",
        "middle_src",
        "top_src",
    ]


def test_objects_at_a_similar_height_share_one_layer() -> None:
    objects = [
        _shelf(),
        _eg_object("left", _SHELF_ID, ObjectType.BOOK, x=-0.2, y=0.0, z=0.50),
        _eg_object("right", _SHELF_ID, ObjectType.BOOK, x=0.2, y=0.0, z=0.51),
    ]

    [layers] = _layers_by_shelf(objects)

    assert len(layers) == 1
    assert {obj.source_id for obj in layers[0].objects} == {"left_src", "right_src"}


def test_a_shelf_whose_own_position_was_not_corrected_yields_no_layers() -> None:
    """
    A layer records its objects' offsets from the shelf's origin, so an uncentred shelf
    position shifts every one of them.

    Such a layer would teach a circuit an arrangement nobody ever built.
    """
    shelf = dataclasses.replace(_shelf(), position_is_mesh_corrected=False)
    book = _eg_object("book_1", _SHELF_ID, ObjectType.BOOK, x=0.0, y=0.0)

    assert _layers_by_shelf([shelf, book]) == []


def test_objects_whose_position_was_not_corrected_are_left_out_of_layers() -> None:
    objects = [
        _shelf(),
        _eg_object("centred", _SHELF_ID, ObjectType.BOOK, x=0.0, y=0.0),
        dataclasses.replace(
            _eg_object("uncentred", _SHELF_ID, ObjectType.BOOK, x=0.2, y=0.0),
            position_is_mesh_corrected=False,
        ),
    ]

    [layers] = _layers_by_shelf(objects)

    assert [obj.source_id for layer in layers for obj in layer.objects] == [
        "centred_src"
    ]


def test_content_orientation_is_stored_relative_to_the_shelfs_content_frame() -> None:
    """
    Contents are spawned inside a corpus built in the shelf's content frame, so a yaw
    stored in absolute terms is double-counted for every rotated shelf.
    """
    objects = [
        _shelf(yaw=90.0),
        _eg_object("book_1", _SHELF_ID, ObjectType.BOOK, x=0.3, y=0.0, yaw=110.0),
    ]

    [layers] = _layers_by_shelf(objects)

    [stored] = layers[0].objects
    assert math.degrees(float(stored.pose.yaw)) == pytest.approx(-70.0)


def test_only_the_requested_object_type_is_extracted() -> None:
    objects = [
        _shelf(),
        _eg_object("book_1", _SHELF_ID, ObjectType.BOOK, x=0.0, y=0.0),
        _eg_object("cup_1", _SHELF_ID, ObjectType.CUP, x=0.1, y=0.0),
    ]

    [layers] = _layers_by_shelf(objects, object_type=ObjectType.BOOK)

    assert [obj.source_id for layer in layers for obj in layer.objects] == [
        "book_1_src"
    ]


def test_every_object_type_is_extracted_by_default() -> None:
    objects = [
        _shelf(),
        _eg_object("book_1", _SHELF_ID, ObjectType.BOOK, x=0.0, y=0.0),
        _eg_object("cup_1", _SHELF_ID, ObjectType.CUP, x=0.1, y=0.0),
    ]

    [layers] = _layers_by_shelf(objects)

    assert {obj.source_id for layer in layers for obj in layer.objects} == {
        "book_1_src",
        "cup_1_src",
    }


# ---------------------------------------------------------------------------
# Layer vertical context -- where a layer sits in its shelf
# ---------------------------------------------------------------------------


def _three_layer_shelf() -> list[EGObject]:
    """
    A shelf whose base is at 0.0 and top at 2.0, holding one book on each of three
    layers at heights 0.2, 0.8 and 1.4, listed top-down.
    """
    return [_shelf()] + [
        _eg_object(name, _SHELF_ID, ObjectType.BOOK, x=0.0, y=0.0, z=height)
        for name, height in [("top", 1.4), ("bottom", 0.2), ("middle", 0.8)]
    ]


def test_layer_height_is_measured_from_the_shelf_meshs_own_base() -> None:
    """
    The shelf's recorded position is its mesh's origin, not its base, so the height a
    layer sits at only follows once the mesh has been measured.
    """
    [layers] = _layers_by_shelf(_three_layer_shelf())

    assert [layer.height_above_shelf_base for layer in layers] == [
        pytest.approx(0.2),
        pytest.approx(0.8),
        pytest.approx(1.4),
    ]


def test_relative_height_places_layers_between_the_shelfs_base_and_top() -> None:
    """
    The fraction is what transfers across shelves of different sizes, so it has to run
    from zero at the base to one at the top.
    """
    [layers] = _layers_by_shelf(_three_layer_shelf())

    assert [layer.relative_height for layer in layers] == [
        pytest.approx(0.1),
        pytest.approx(0.4),
        pytest.approx(0.7),
    ]


def test_vertical_clearance_reaches_the_next_layer_up() -> None:
    [layers] = _layers_by_shelf(_three_layer_shelf())

    assert layers[0].vertical_clearance == pytest.approx(0.6)
    assert layers[1].vertical_clearance == pytest.approx(0.6)


def test_the_topmost_layers_clearance_reaches_the_shelfs_top() -> None:
    """
    Nothing stands above the topmost layer, so its clearance is the room left under the
    shelf's own ceiling -- which is what decides what still fits.
    """
    [layers] = _layers_by_shelf(_three_layer_shelf())

    assert layers[-1].vertical_clearance == pytest.approx(0.6)


def test_a_shelf_of_no_measurable_height_reads_as_sitting_at_its_base() -> None:
    """
    A degenerate mesh must not divide by zero; the absolute height stays meaningful even
    when the fraction cannot be.
    """
    objects = [
        _shelf(),
        _eg_object("book_1", _SHELF_ID, ObjectType.BOOK, x=0.0, y=0.0, z=1.0),
    ]

    shelves = shelves_with_layers(
        objects,
        {_SHELF_SOURCE_ID: VerticalExtent(bottom=0.0, top=0.0)},
        {_SHELF_ID},
        MeshMeasurements(source_id_to_path={}),
    )

    [layer] = shelves[0].layers
    assert layer.relative_height == pytest.approx(0.0)
    assert layer.height_above_shelf_base == pytest.approx(0.0)


def test_a_shelf_whose_mesh_was_never_measured_is_skipped() -> None:
    """
    Without the shelf's real base and top its layers' heights would be guesswork, so it
    contributes nothing rather than something invented.
    """
    objects = [
        _shelf(),
        _eg_object("book_1", _SHELF_ID, ObjectType.BOOK, x=0.0, y=0.0),
    ]

    assert (
        shelves_with_layers(
            objects,
            {},
            {_SHELF_ID},
            MeshMeasurements(source_id_to_path={}),
        )
        == []
    )


def test_a_shelf_keeps_its_own_pose_and_measured_height() -> None:
    """
    Keeping the shelf, not just loose layers, is what preserves which layers belong
    together and in what order.
    """
    [shelf] = shelves_with_layers(
        _three_layer_shelf(),
        {_SHELF_SOURCE_ID: _SHELF_EXTENT},
        {_SHELF_ID},
        MeshMeasurements(source_id_to_path={}),
    )

    assert shelf.scale.z == pytest.approx(2.0)
    assert len(shelf.layers) == 3
    assert [obj.source_id for layer in shelf.layers for obj in layer.objects] == [
        "bottom_src",
        "middle_src",
        "top_src",
    ]


def test_a_shelfs_theme_is_the_object_type_its_objects_have_the_most_of() -> None:
    """
    A shelf's theme is derived from what is actually placed on it, so two books and one
    bottle must make the shelf book-themed -- and every layer and object on it must
    carry that same theme, since it is denormalized onto all three.
    """
    objects = [_shelf()] + [
        _eg_object("book_1", _SHELF_ID, ObjectType.BOOK, x=0.0, y=0.0, z=0.5),
        _eg_object("book_2", _SHELF_ID, ObjectType.BOOK, x=0.2, y=0.0, z=0.5),
        _eg_object("bottle_1", _SHELF_ID, ObjectType.BOTTLE, x=0.4, y=0.0, z=0.5),
    ]

    [shelf] = shelves_with_layers(
        objects,
        {_SHELF_SOURCE_ID: _SHELF_EXTENT},
        {_SHELF_ID},
        MeshMeasurements(source_id_to_path={}),
    )

    assert shelf.theme_dominant_type is ObjectType.BOOK
    assert {layer.theme_dominant_type for layer in shelf.layers} == {ObjectType.BOOK}
    assert {
        obj.theme_dominant_type for layer in shelf.layers for obj in layer.objects
    } == {ObjectType.BOOK}


def test_a_tied_theme_breaks_alphabetically_by_type_value() -> None:
    """
    A tie between equally-frequent types must resolve the same way every time rather
    than depend on iteration order -- broken here by the type's own, ascending value
    ("book" < "bottle").
    """
    objects = [_shelf()] + [
        _eg_object("bottle_1", _SHELF_ID, ObjectType.BOTTLE, x=0.0, y=0.0, z=0.5),
        _eg_object("book_1", _SHELF_ID, ObjectType.BOOK, x=0.2, y=0.0, z=0.5),
    ]

    [shelf] = shelves_with_layers(
        objects,
        {_SHELF_SOURCE_ID: _SHELF_EXTENT},
        {_SHELF_ID},
        MeshMeasurements(source_id_to_path={}),
    )

    assert shelf.theme_dominant_type is ObjectType.BOOK


# ---------------------------------------------------------------------------
# ShelfContents -- keeping only what layer extraction reads
# ---------------------------------------------------------------------------


def test_shelf_ids_come_from_the_classified_types_of_the_raw_objects(
    session: Session,
) -> None:
    """
    Which raw objects are shelves has to be settled before the dataset is read in full,
    since that is what tells the main pass whether an object may be dropped the moment
    it has been written.
    """
    session.add_all(
        [
            _sage10k_object(object_id="shelf_1", raw_type="bookshelf1"),
            _sage10k_object(object_id="book_1", raw_type="book2"),
        ]
    )
    session.commit()

    contents = ShelfContents.from_raw_objects(session, ShelfMembershipClassifier())

    assert contents.shelf_ids == {"shelf_1"}


def test_shelves_and_the_objects_standing_on_them_are_kept() -> None:
    shelf = _shelf()
    on_shelf = _eg_object("book_1", _SHELF_ID, ObjectType.BOOK, x=0.0, y=0.0)
    elsewhere = _eg_object("chair_1", "floor", ObjectType.CHAIR, x=5.0, y=5.0)
    contents = ShelfContents(shelf_ids={_SHELF_ID})

    for processed_object in [shelf, on_shelf, elsewhere]:
        contents.collect(processed_object)

    assert contents.objects == [shelf, on_shelf]


def test_a_shelf_like_object_is_not_counted_as_another_shelfs_content() -> None:
    """
    An object that is itself classified as a shelf-like parent must not also be
    counted as ordinary content standing on a different shelf: the raw dataset
    records a small piece of shelf-like furniture placed on a bigger one this way,
    and treating it as passive content teaches the circuit that shelves commonly
    hold other shelves.
    """
    shelf = _shelf()
    nested_shelf_like_object = _eg_object(
        "small_shelf_1", _SHELF_ID, ObjectType.SHELF, x=0.0, y=0.0
    )
    book = _eg_object("book_1", _SHELF_ID, ObjectType.BOOK, x=0.3, y=0.0)
    shelf_ids = {_SHELF_ID, "small_shelf_1"}

    [extracted_shelf] = shelves_with_layers(
        [shelf, nested_shelf_like_object, book],
        {_SHELF_SOURCE_ID: _SHELF_EXTENT},
        shelf_ids,
        MeshMeasurements(source_id_to_path={}),
    )

    content_source_ids = {
        obj.source_id for layer in extracted_shelf.layers for obj in layer.objects
    }
    assert content_source_ids == {"book_1_src"}


def test_extraction_from_the_kept_objects_matches_extraction_from_all_of_them() -> None:
    """
    Holding back only shelves and their contents is what keeps the pipeline's memory
    bounded, so it must leave the extracted shelves exactly as they would have been had
    the whole dataset been kept.
    """
    every_object = _three_layer_shelf() + [
        _eg_object("chair_1", "floor", ObjectType.CHAIR, x=5.0, y=5.0),
        _eg_object("cup_1", "table_1", ObjectType.CUP, x=6.0, y=6.0),
    ]
    contents = ShelfContents(shelf_ids={_SHELF_ID})
    for processed_object in every_object:
        contents.collect(processed_object)

    vertical_extents = {_SHELF_SOURCE_ID: _SHELF_EXTENT}
    measurements = MeshMeasurements(source_id_to_path={})
    # Compared through to_json() rather than == : EGObject2D.pose is a Pose2D, which
    # has no meaningful equality (it falls back to identity), so two independently
    # built shelf trees would never compare equal even when numerically identical.
    assert [
        shelf.to_json()
        for shelf in shelves_with_layers(
            contents.objects,
            vertical_extents,
            contents.shelf_ids,
            measurements,
        )
    ] == [
        shelf.to_json()
        for shelf in shelves_with_layers(
            every_object,
            vertical_extents,
            contents.shelf_ids,
            measurements,
        )
    ]


def test_kept_shelves_supply_the_vertical_extents_their_layers_need(
    tmp_path: Path,
) -> None:
    """
    Only the shelves' own meshes have to be measured; measuring every object's would
    load meshes whose reach nothing reads.
    """
    scene_directory = _cache_off_center_mesh(tmp_path, _SHELF_SOURCE_ID)
    contents = ShelfContents(shelf_ids={_SHELF_ID})
    for processed_object in _three_layer_shelf():
        contents.collect(processed_object)

    extents = contents.vertical_extents(
        MeshMeasurements(source_id_to_path={_SHELF_SOURCE_ID: scene_directory})
    )

    assert set(extents) == {_SHELF_SOURCE_ID}


# ---------------------------------------------------------------------------
# object_type_height_profiles -- which types are kept high, which low
# ---------------------------------------------------------------------------


def _layer_at(relative_height: float, *objects: EGObject2D) -> EGShelfLayer:
    return EGShelfLayer(
        objects=list(objects),
        height_above_shelf_base=relative_height * 2.0,
        relative_height=relative_height,
        vertical_clearance=0.3,
        theme_dominant_type=ObjectType.BOOK,
    )


def test_height_profile_averages_where_a_type_was_found() -> None:
    layers = [
        _layer_at(0.2, _object_2d(ObjectType.BOOK, "book_1")),
        _layer_at(0.4, _object_2d(ObjectType.BOOK, "book_2")),
    ]

    [profile] = object_type_height_profiles(layers)

    assert profile.object_type == ObjectType.BOOK
    assert profile.observation_count == 2
    assert profile.mean_relative_height == pytest.approx(0.3)
    assert profile.mean_height_above_shelf_base == pytest.approx(0.6)


def test_height_profile_separates_types_kept_high_from_types_kept_low() -> None:
    """
    The whole point of the profile: a robot holding a book has to be able to
    tell that books belong lower than display pieces.
    """
    layers = [
        _layer_at(0.1, _object_2d(ObjectType.BOOK, "book_1")),
        _layer_at(0.2, _object_2d(ObjectType.BOOK, "book_2")),
        _layer_at(0.9, _object_2d(ObjectType.VASE, "vase_1")),
    ]

    profiles = {
        profile.object_type: profile for profile in object_type_height_profiles(layers)
    }

    assert profiles[ObjectType.BOOK].mean_relative_height < (
        profiles[ObjectType.VASE].mean_relative_height
    )


def test_height_profile_counts_every_object_of_a_type_on_a_layer() -> None:
    layers = [
        _layer_at(
            0.5,
            _object_2d(ObjectType.BOOK, "book_1"),
            _object_2d(ObjectType.BOOK, "book_2"),
        )
    ]

    [profile] = object_type_height_profiles(layers)

    assert profile.observation_count == 2


def test_height_profiles_are_empty_without_layers() -> None:
    assert object_type_height_profiles([]) == []


# ---------------------------------------------------------------------------
# Extraction and spawning must be inverses
# ---------------------------------------------------------------------------


def _cache_book_mesh(tmp_path: Path) -> None:
    """
    Copy a real textured mesh into *tmp_path* under the ``book_src`` source id, so a
    spawned shelf has geometry to place.
    """
    resources_root = (
        Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
    )
    objects_directory = tmp_path / "objects"
    objects_directory.mkdir()
    shutil.copy(resources_root / "chair.ply", objects_directory / "book_src.ply")
    shutil.copy(
        resources_root / "chair_texture.png",
        objects_directory / "book_src_texture.png",
    )


def test_extracted_contents_spawn_back_at_their_original_world_pose(
    tmp_path: Path,
) -> None:
    """
    Extraction and spawning have to be inverses: an object at a known world pose on a
    rotated shelf must come back to that pose once its extracted, shelf-local pose is
    spawned again.

    Storing the pose in the shelf's frame is only half the contract -- the spawn side
    re-applies the shelf's rotation and maps the stored axes onto the corpus frame.
    """
    _cache_book_mesh(tmp_path)
    shelf_world_x, shelf_world_y, shelf_yaw = 10.0, 5.0, 90.0
    book_world_x, book_world_y = 10.3, 5.2
    shelf = _eg_object(
        _SHELF_ID,
        place_id="floor",
        object_type=ObjectType.SHELF,
        x=shelf_world_x,
        y=shelf_world_y,
        z=1.0,
        yaw=shelf_yaw,
        width=1.0,
        length=1.0,
    )
    book = _eg_object(
        "book_1",
        place_id=_SHELF_ID,
        object_type=ObjectType.BOOK,
        x=book_world_x,
        y=book_world_y,
        yaw=110.0,
        source_id="book_src",
    )

    [layers] = _layers_by_shelf([shelf, book])
    spawned = EGShelf(
        scale=Scale(x=1.0, y=1.0, z=2.0),
        layers=layers,
        source_ids=[
            MeshCandidate(
                scene_dir=tmp_path, source_id="book_src", object_type=ObjectType.BOOK
            )
        ],
        theme_dominant_type=ObjectType.BOOK,
    )
    world, root = _empty_world()
    spawned.spawn(world, parent=root)
    _move_shelf_to(spawned, shelf_world_x, shelf_world_y, shelf_yaw)

    body = spawned.layers[0].objects[0].annotation
    position = body.global_pose.to_position().to_np()
    assert position[0] == pytest.approx(book_world_x, abs=1e-6)
    assert position[1] == pytest.approx(book_world_y, abs=1e-6)


def test_extracted_contents_spawn_within_the_layer_footprint(tmp_path: Path) -> None:
    """
    An object offset along the shelf's wide face must spawn inside the corpus footprint
    on both axes.

    Were that face offset mapped onto the corpus's shallow depth axis, the object would
    protrude front and back.
    """
    _cache_book_mesh(tmp_path)
    shelf_depth, shelf_face = 0.3, 1.0
    shelf = _eg_object(
        _SHELF_ID,
        place_id="floor",
        object_type=ObjectType.SHELF,
        x=0.0,
        y=0.0,
        z=1.0,
        width=shelf_face,
        length=shelf_depth,
    )
    # At the shelf's zero yaw its wide face lies along world x, so a world-x
    # offset is a face offset: well within the face, far outside the depth.
    book = _eg_object(
        "book_1",
        place_id=_SHELF_ID,
        object_type=ObjectType.BOOK,
        x=0.4,
        y=0.0,
        source_id="book_src",
    )

    [layers] = _layers_by_shelf([shelf, book])
    spawned = EGShelf(
        scale=Scale(x=shelf_depth, y=shelf_face, z=2.0),
        layers=layers,
        source_ids=[
            MeshCandidate(
                scene_dir=tmp_path, source_id="book_src", object_type=ObjectType.BOOK
            )
        ],
        theme_dominant_type=ObjectType.BOOK,
    )
    world, root = _empty_world()
    spawned.spawn(world, parent=root)

    body = spawned.layers[0].objects[0].annotation
    corpus_x, corpus_y = body.parent_connection.origin.to_position().to_np()[:2]
    assert abs(corpus_x) <= shelf_depth / 2
    assert abs(corpus_y) <= shelf_face / 2


# ---------------------------------------------------------------------------
# object_type_affinities -- which types share a layer, and how they sit
# ---------------------------------------------------------------------------


def test_affinity_counts_every_pair_sharing_a_layer() -> None:
    layers = [
        EGShelfLayer(
            objects=[
                _object_2d(ObjectType.BOOK, "book_1"),
                _object_2d(ObjectType.CUP, "cup_1"),
            ],
            theme_dominant_type=ObjectType.BOOK,
        ),
        EGShelfLayer(
            objects=[
                _object_2d(ObjectType.BOOK, "book_2"),
                _object_2d(ObjectType.CUP, "cup_2"),
            ],
            theme_dominant_type=ObjectType.BOOK,
        ),
    ]

    [affinity] = object_type_affinities(layers)

    assert affinity.co_occurrence_count == 2


def test_affinity_pairs_are_stored_in_canonical_order() -> None:
    """
    A pair must land on the same row whichever order the two objects happen to appear
    in, or the same relationship is split across two half-counted rows.
    """
    layers = [
        EGShelfLayer(
            objects=[
                _object_2d(ObjectType.CUP, "cup_1"),
                _object_2d(ObjectType.BOOK, "book_1"),
            ],
            theme_dominant_type=ObjectType.BOOK,
        )
    ]

    [affinity] = object_type_affinities(layers)

    assert affinity.object_type_a == ObjectType.BOOK
    assert affinity.object_type_b == ObjectType.CUP


def test_affinity_mean_offset_points_from_the_first_type_to_the_second() -> None:
    layers = [
        EGShelfLayer(
            objects=[
                _object_2d(ObjectType.CUP, "cup_1", x=0.3, y=0.1),
                _object_2d(ObjectType.BOOK, "book_1", x=0.1, y=0.1),
            ],
            theme_dominant_type=ObjectType.BOOK,
        )
    ]

    [affinity] = object_type_affinities(layers)

    assert affinity.mean_relative_offset.x == pytest.approx(0.2)
    assert affinity.mean_relative_offset.y == pytest.approx(0.0)


def test_affinity_averages_the_offset_over_every_observed_pair() -> None:
    layers = [
        EGShelfLayer(
            objects=[
                _object_2d(ObjectType.BOOK, "book_1", x=0.0, y=0.0),
                _object_2d(ObjectType.CUP, "cup_1", x=0.2, y=0.0),
            ],
            theme_dominant_type=ObjectType.BOOK,
        ),
        EGShelfLayer(
            objects=[
                _object_2d(ObjectType.BOOK, "book_2", x=0.0, y=0.0),
                _object_2d(ObjectType.CUP, "cup_2", x=0.4, y=0.0),
            ],
            theme_dominant_type=ObjectType.BOOK,
        ),
    ]

    [affinity] = object_type_affinities(layers)

    assert affinity.mean_relative_offset.x == pytest.approx(0.3)


def test_affinity_includes_pairs_of_the_same_type() -> None:
    """
    How far apart two books usually sit is as useful for placement as which other types
    accompany them.
    """
    layers = [
        EGShelfLayer(
            objects=[
                _object_2d(ObjectType.BOOK, "book_1", x=0.0, y=0.0),
                _object_2d(ObjectType.BOOK, "book_2", x=0.1, y=0.0),
            ],
            theme_dominant_type=ObjectType.BOOK,
        )
    ]

    [affinity] = object_type_affinities(layers)

    assert affinity.object_type_a == ObjectType.BOOK
    assert affinity.object_type_b == ObjectType.BOOK
    assert affinity.co_occurrence_count == 1


def test_affinity_is_empty_for_layers_holding_a_single_object() -> None:
    layers = [
        EGShelfLayer(
            objects=[_object_2d(ObjectType.BOOK, "book_1")],
            theme_dominant_type=ObjectType.BOOK,
        )
    ]

    assert object_type_affinities(layers) == []
