from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from experiments.orm.ormatic_interface import (
    Base,
    EGObjectDAO,
    EGPositionDAO,
    EGRoomDAO,
    EGRotationDAO,
    EGScaleDAO,
)
from experiments.scene_generation_experiments.collision_resolution import (
    build_free_room_floor_query,
)
from experiments.scene_generation_experiments.room_floor_sampling import (
    SampledRoomComposition,
)
from experiments.scene_generation_experiments.utils import min_samples_per_leaf_for
from experiments.scene_generation_experiments.room_generation import (
    _extract_room_floor_layouts,
)
from experiments.scene_generation_experiments.utils import (
    build_cached_mesh_pool,
    objects_for_rooms,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.feature_extraction.aggregations import (
    compute_aggregation_statistics,
)
from krrood.ormatic.utils import create_engine
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGFloorPiece,
    EGWallRelativePose,
    EGRoomFloorLayout,
    EGRotation,
    EGScale,
    ObjectType,
    PlaceId,
    RoomType,
)


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    database_session = Session(engine)
    yield database_session
    database_session.close()


def _make_object(
    object_id: str,
    room_id: str,
    object_type: ObjectType,
    x: float,
    y: float,
    place_id: str = "floor",
    height: float = 1.0,
) -> EGObjectDAO:
    return EGObjectDAO(
        id=object_id,
        room_id=room_id,
        place_id=place_id,
        source_id=f"{object_id}_src",
        object_type=object_type,
        scale=EGScaleDAO(height=height, length=0.8, width=0.8),
        position=EGPositionDAO(x=x, y=y, z=height / 2),
        orientation=EGRotationDAO(x=0.0, y=0.0, z=0.0),
    )


def _make_room(
    room_id: str,
    room_type: RoomType = RoomType.KITCHEN,
    width: float = 6.0,
    length: float = 8.0,
) -> EGRoomDAO:
    return EGRoomDAO(
        id=room_id,
        room_type=room_type,
        scale=EGScaleDAO(height=2.7, length=length, width=width),
        position=EGPositionDAO(x=0.0, y=0.0, z=0.0),
    )


def test_floor_pieces_are_grouped_per_room(session: Session) -> None:
    """
    Floor-resting pieces must be grouped into one layout per room, so each
    room's floor arrangement is a single training instance.
    """
    session.add_all(
        [
            _make_room("room_1"),
            _make_room("room_2"),
            _make_object("shelf_1", "room_1", ObjectType.SHELF, x=0.0, y=0.0),
            _make_object("table_1", "room_1", ObjectType.TABLE, x=2.0, y=0.0),
            _make_object("shelf_2", "room_2", ObjectType.SHELF, x=0.0, y=0.0),
        ]
    )
    session.commit()

    layouts, _ = _extract_room_floor_layouts(session, RoomType.KITCHEN)

    assert len(layouts) == 2
    piece_counts = sorted(len(layout.pieces) for layout in layouts)
    assert piece_counts == [1, 2]


def test_pieces_placed_on_other_pieces_are_skipped(session: Session) -> None:
    """
    A piece that references another piece via its ``place_id`` -- e.g. a table
    on another table -- does not rest on the floor and must be excluded.
    """
    session.add_all(
        [
            _make_room("room_1"),
            _make_object("table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0),
            _make_object(
                "lamp_1", "room_1", ObjectType.LAMP, x=0.1, y=0.0, place_id="table_1"
            ),
        ]
    )
    session.commit()

    layouts, _ = _extract_room_floor_layouts(session, RoomType.KITCHEN)

    assert len(layouts) == 1
    assert [piece.object_type for piece in layouts[0].pieces] == [ObjectType.TABLE]


def test_room_footprint_comes_from_the_stored_room_not_from_its_pieces(
    session: Session,
) -> None:
    """
    The footprint must be the room's own recorded size. Deriving it from the
    pieces made the room a function of whichever furniture happened to load, so
    a sparsely populated room collapsed to a couple of metres across even when
    the real room was large.
    """
    session.add_all(
        [
            _make_room("room_1", width=6.0, length=8.0),
            _make_object("shelf_1", "room_1", ObjectType.SHELF, x=2.9, y=3.9),
            _make_object("shelf_2", "room_1", ObjectType.SHELF, x=3.1, y=4.1),
        ]
    )
    session.commit()

    layouts, _ = _extract_room_floor_layouts(session, RoomType.KITCHEN)

    assert layouts[0].scale.width == pytest.approx(6.0)
    assert layouts[0].scale.length == pytest.approx(8.0)
    assert layouts[0].scale.height == pytest.approx(2.7)


def test_piece_positions_are_centred_on_the_stored_room_footprint(
    session: Session,
) -> None:
    """
    Stored piece positions are room-local with the room's lower-left corner at
    the origin, so re-centring must subtract half the room's own extent -- not
    the centre of the pieces' bounding box.

    Checked through the wall-relative pose the layout actually stores, so this
    also covers the conversion round-tripping back to the original coordinates.
    """
    session.add_all(
        [
            _make_room("room_1", width=6.0, length=8.0),
            _make_object("centre", "room_1", ObjectType.SHELF, x=3.0, y=4.0),
            _make_object("corner", "room_1", ObjectType.LAMP, x=0.0, y=0.0),
        ]
    )
    session.commit()

    layouts, _ = _extract_room_floor_layouts(session, RoomType.KITCHEN)

    recovered = {
        piece.object_type: piece.pose.to_absolute_pose(layouts[0].scale)[:2]
        for piece in layouts[0].pieces
    }
    assert recovered[ObjectType.SHELF] == pytest.approx((0.0, 0.0))
    assert recovered[ObjectType.LAMP] == pytest.approx((-3.0, -4.0))


def test_only_rooms_of_the_requested_type_are_extracted(session: Session) -> None:
    """
    Circuits are fitted per room type, so pooling every category back together
    at extraction time would defeat the whole point.
    """
    session.add_all(
        [
            _make_room("kitchen_1", room_type=RoomType.KITCHEN),
            _make_room("warehouse_1", room_type=RoomType.WAREHOUSE),
            _make_object("piece_1", "kitchen_1", ObjectType.SHELF, x=1.0, y=1.0),
            _make_object("piece_2", "warehouse_1", ObjectType.CABINET, x=1.0, y=1.0),
        ]
    )
    session.commit()

    layouts, _ = _extract_room_floor_layouts(session, RoomType.KITCHEN)

    assert len(layouts) == 1
    assert [piece.object_type for piece in layouts[0].pieces] == [ObjectType.SHELF]


def test_room_cap_selects_whole_rooms_without_truncating_their_pieces(
    session: Session,
) -> None:
    """
    Capping how many rooms are sampled for training must never truncate a
    selected room's own piece set: a room's floor pieces used to be cut off
    by a flat row-count limit on the underlying object query, leaving most
    "rooms" with only 1-2 of their true pieces represented.
    """
    piece_count_per_room = 30
    for room_index in range(3):
        session.add(_make_room(f"room_{room_index}"))
        session.add_all(
            [
                _make_object(
                    f"room{room_index}_piece{piece_index}",
                    f"room_{room_index}",
                    ObjectType.SHELF,
                    x=float(piece_index),
                    y=0.0,
                )
                for piece_index in range(piece_count_per_room)
            ]
        )
    session.commit()

    layouts, _ = _extract_room_floor_layouts(session, RoomType.KITCHEN, room_count=2)

    assert len(layouts) == 2
    for layout in layouts:
        assert len(layout.pieces) == piece_count_per_room


def test_objects_for_rooms_returns_every_row_without_a_cap(session: Session) -> None:
    """
    A 50000-row cap on this query silently truncated training rooms, cutting the
    median floor-piece count from 22 to 9 and making generated rooms both tiny
    and nearly empty.
    """
    room_count = 200
    pieces_per_room = 30
    for room_index in range(room_count):
        session.add_all(
            [
                _make_object(
                    f"room{room_index}_piece{piece_index}",
                    f"room_{room_index}",
                    ObjectType.SHELF,
                    x=float(piece_index),
                    y=0.0,
                )
                for piece_index in range(pieces_per_room)
            ]
        )
    session.commit()

    loaded = objects_for_rooms(session, [f"room_{i}" for i in range(room_count)])

    assert len(loaded) == room_count * pieces_per_room


def test_objects_for_rooms_can_restrict_to_one_place_id(session: Session) -> None:
    """
    The room pipeline only needs floor pieces, but the shelf and table
    extractors need the rest, so the filter has to be opt-in rather than
    baked into the query.
    """
    session.add_all(
        [
            _make_object("floor_piece", "room_1", ObjectType.SHELF, x=0.0, y=0.0),
            _make_object(
                "on_shelf", "room_1", ObjectType.BOOK, x=0.0, y=0.0, place_id="floor_piece"
            ),
        ]
    )
    session.commit()

    assert len(objects_for_rooms(session, ["room_1"])) == 2
    floor_only = objects_for_rooms(session, ["room_1"], place_id=PlaceId.FLOOR)
    assert [obj.id for obj in floor_only] == ["floor_piece"]


def _synthetic_floor_layout(
    room_index: int,
    piece_count: int,
    rng: random.Random,
    scale: EGScale | None = None,
) -> EGRoomFloorLayout:
    room_scale = scale or EGScale(width=10.0, length=10.0, height=2.5)
    pieces = [
        EGFloorPiece(
            object_type=rng.choice(
                [ObjectType.TABLE, ObjectType.CHAIR, ObjectType.SHELF]
            ),
            scale=EGScale(
                width=rng.uniform(0.3, 1.5),
                length=rng.uniform(0.3, 1.5),
                height=rng.uniform(0.3, 2.0),
            ),
            pose=EGWallRelativePose.from_absolute_pose(
                rng.uniform(-room_scale.width / 2, room_scale.width / 2),
                rng.uniform(-room_scale.length / 2, room_scale.length / 2),
                rng.uniform(0.0, 360.0),
                room_scale,
            ),
        )
        for _ in range(piece_count)
    ]
    return EGRoomFloorLayout(scale=room_scale, pieces=pieces)


def test_min_samples_per_leaf_fraction_bounds_the_pieces_circuit_size() -> None:
    """
    Each floor piece carries near-unique ``id`` and ``source_id`` values, so
    fitting the "pieces" RSPN with the library's default ``min_samples_per_leaf``
    of one lets its circuit grow roughly one leaf per training piece --
    thousands of nodes even for a modest training set. Grounding deep-copies
    that circuit once per sampled piece, which exhausted memory before a room
    could be sampled. ``min_samples_per_leaf_for(sum(len(l.pieces) for l in layouts))`` must keep the fitted
    circuit small enough that grounding stays cheap.
    """
    rng = random.Random(0)
    layouts = [_synthetic_floor_layout(i, rng.randint(3, 8), rng) for i in range(60)]
    daos = [to_dao(layout) for layout in layouts]

    default_model = RelationalProbabilisticCircuit(EGRoomFloorLayout).fit(daos)
    bounded_model = RelationalProbabilisticCircuit(
        EGRoomFloorLayout, min_samples_per_leaf=min_samples_per_leaf_for(sum(len(l.pieces) for l in layouts))
    ).fit(daos)

    default_piece_nodes = len(
        default_model.exchangeable_distribution_templates[
            "pieces"
        ].template_distribution.class_probabilistic_circuit.nodes()
    )
    bounded_piece_nodes = len(
        bounded_model.exchangeable_distribution_templates[
            "pieces"
        ].template_distribution.class_probabilistic_circuit.nodes()
    )
    assert bounded_piece_nodes < default_piece_nodes / 2


def _fitted_layout_model() -> RelationalProbabilisticCircuit:
    """
    Fit a room-floor circuit on layouts whose footprints genuinely vary, so the
    room-geometry aggregations have a distribution to be conditioned on.
    """
    rng = random.Random(0)
    layouts = [
        _synthetic_floor_layout(
            room_index,
            rng.randint(3, 8),
            rng,
            scale=EGScale(
                width=rng.uniform(3.0, 12.0),
                length=rng.uniform(3.0, 12.0),
                height=2.7,
            ),
        )
        for room_index in range(60)
    ]
    return RelationalProbabilisticCircuit(
        EGRoomFloorLayout, min_samples_per_leaf=min_samples_per_leaf_for(sum(len(l.pieces) for l in layouts))
    ).fit([to_dao(layout) for layout in layouts])


def test_room_geometry_aggregations_become_latent_variables_of_the_pieces_template() -> None:
    """
    Aggregations are the only channel carrying room-level context into the
    piece circuit, so ``floor_area`` and ``aspect_ratio`` must surface as latent
    variables of the ``pieces`` template. Without them a piece's position is
    conditioned only on how many pieces the room holds, never on how big or what
    shape it is.
    """
    template = _fitted_layout_model().exchangeable_distribution_templates["pieces"]

    latent_names = {variable.name for variable in template.latent_variables}
    assert "EGRoomFloorLayoutAggregations.floor_area()" in latent_names
    assert "EGRoomFloorLayoutAggregations.aspect_ratio()" in latent_names


def test_room_geometry_aggregations_are_determined_by_a_footprint_fixed_query() -> None:
    """
    Both statistics must be computable from the query itself, so grounding
    conditions on them directly instead of integrating them out via Monte-Carlo.
    Integration would marginalise the latents out of the class circuit, leaving
    the sampled room footprint statistically decoupled from the piece positions
    it produced -- reintroducing the very defect these aggregations exist to fix.
    """
    model = _fitted_layout_model()
    template = model.exchangeable_distribution_templates["pieces"]
    query = build_free_room_floor_query(
        SampledRoomComposition(
            object_types=[ObjectType.SHELF] * 4,
            scale=EGScale(width=6.0, length=3.0, height=2.7),
        )
    )
    query.resolve()

    statistics = compute_aggregation_statistics(
        query.construct_instance(),
        model.feature_extractor.exchangeable_features["pieces"],
        template.latent_variables,
    )

    determined = {variable.name: value for variable, value in statistics.items()}
    assert determined["EGRoomFloorLayoutAggregations.floor_area()"] == pytest.approx(18.0)
    assert determined["EGRoomFloorLayoutAggregations.aspect_ratio()"] == pytest.approx(2.0)

    undetermined = [
        variable for variable in template.latent_variables if variable not in statistics
    ]
    assert undetermined == []


def test_scene_mesh_pool_uses_only_the_cache_without_a_downloader() -> None:
    """
    Building the scene mesh pool without a downloader must not trigger any
    download, so the demo stays fast for iterative testing.
    """
    with patch(
        "experiments.scene_generation_experiments.utils"
        ".download_meshes_for_floor_object_types"
    ) as download, patch(
        "experiments.scene_generation_experiments.utils.build_source_id_to_path",
        return_value={},
    ), patch(
        "experiments.scene_generation_experiments.utils.load_objects_with_cached_meshes",
        return_value=[],
    ) as load_cached:
        result = build_cached_mesh_pool(MagicMock())

    download.assert_not_called()
    load_cached.assert_called_once()
    assert result == []


def test_scene_mesh_pool_tops_up_the_cache_when_given_a_downloader() -> None:
    """
    A downloader must broaden the pool by downloading floor-object meshes before
    the cached objects are loaded.
    """
    session = MagicMock()
    downloader = MagicMock()

    with patch(
        "experiments.scene_generation_experiments.utils"
        ".download_meshes_for_floor_object_types"
    ) as download, patch(
        "experiments.scene_generation_experiments.utils.build_source_id_to_path",
        return_value={},
    ), patch(
        "experiments.scene_generation_experiments.utils.load_objects_with_cached_meshes",
        return_value=[],
    ):
        build_cached_mesh_pool(session, downloader)

    download.assert_called_once_with(session, downloader)
