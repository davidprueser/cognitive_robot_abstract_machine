from __future__ import annotations

import random

import pytest
from sqlalchemy.orm import Session

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from experiments.orm.ormatic_interface import (
    Base,
    EGObjectDAO,
    EGPositionDAO,
    EGRotationDAO,
    EGScaleDAO,
)
from experiments.scene_generation_experiments.room_generation import (
    _MIN_SAMPLES_PER_LEAF_FRACTION,
    _extract_room_floor_layouts,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGPoint2D,
    EGRoomFloorLayout,
    EGRotation,
    EGScale,
    ObjectType,
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


def test_floor_pieces_are_grouped_per_room(session: Session) -> None:
    """
    Floor-resting pieces must be grouped into one layout per room, so each
    room's floor arrangement is a single training instance.
    """
    session.add_all(
        [
            _make_object("shelf_1", "room_1", ObjectType.SHELF, x=0.0, y=0.0),
            _make_object("table_1", "room_1", ObjectType.TABLE, x=2.0, y=0.0),
            _make_object("shelf_2", "room_2", ObjectType.SHELF, x=0.0, y=0.0),
        ]
    )
    session.commit()

    layouts, _ = _extract_room_floor_layouts(session)

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
            _make_object("table_1", "room_1", ObjectType.TABLE, x=0.0, y=0.0),
            _make_object(
                "table_2", "room_1", ObjectType.TABLE, x=0.1, y=0.0, place_id="table_1"
            ),
        ]
    )
    session.commit()

    layouts, _ = _extract_room_floor_layouts(session)

    assert len(layouts) == 1
    assert len(layouts[0].pieces) == 1
    assert layouts[0].pieces[0].id == "table_1"


def test_piece_positions_are_centred_on_the_room_footprint(session: Session) -> None:
    """
    Piece positions must be re-expressed relative to the footprint centre, so the
    layout is learnable independent of where the room sits in world coordinates.
    """
    session.add_all(
        [
            _make_object("shelf_1", "room_1", ObjectType.SHELF, x=2.0, y=4.0),
            _make_object("shelf_2", "room_1", ObjectType.SHELF, x=4.0, y=8.0),
        ]
    )
    session.commit()

    layouts, _ = _extract_room_floor_layouts(session)

    positions = {piece.id: piece.position for piece in layouts[0].pieces}
    assert positions["shelf_1"] == EGPoint2D(x=-1.0, y=-2.0)
    assert positions["shelf_2"] == EGPoint2D(x=1.0, y=2.0)


def test_room_footprint_spans_the_pieces_bounding_box_with_margin(
    session: Session,
) -> None:
    """
    The learned room footprint must cover the pieces' bounding box plus a margin,
    so the room is not sized flush to its furniture.
    """
    session.add_all(
        [
            _make_object("shelf_1", "room_1", ObjectType.SHELF, x=0.0, y=0.0),
            _make_object("shelf_2", "room_1", ObjectType.SHELF, x=3.0, y=5.0),
        ]
    )
    session.commit()

    layouts, _ = _extract_room_floor_layouts(session)

    assert layouts[0].scale.width == pytest.approx(4.0)
    assert layouts[0].scale.length == pytest.approx(6.0)


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

    layouts, _ = _extract_room_floor_layouts(session, room_count=2)

    assert len(layouts) == 2
    for layout in layouts:
        assert len(layout.pieces) == piece_count_per_room


def _synthetic_floor_layout(room_index: int, piece_count: int, rng: random.Random) -> EGRoomFloorLayout:
    pieces = [
        EGObject2D(
            id=f"room{room_index}_piece{piece_index}",
            room_id=f"room{room_index}",
            place_id="floor",
            object_type=rng.choice(
                [ObjectType.TABLE, ObjectType.CHAIR, ObjectType.SHELF]
            ),
            scale=EGScale(
                width=rng.uniform(0.3, 1.5),
                length=rng.uniform(0.3, 1.5),
                height=rng.uniform(0.3, 2.0),
            ),
            position=EGPoint2D(x=rng.uniform(-5.0, 5.0), y=rng.uniform(-5.0, 5.0)),
            orientation=EGRotation(x=0.0, y=0.0, z=rng.uniform(0.0, 360.0)),
            source_id=f"mesh_{rng.randint(0, 40)}",
        )
        for piece_index in range(piece_count)
    ]
    return EGRoomFloorLayout(
        scale=EGScale(width=10.0, length=10.0, height=2.5), pieces=pieces
    )


def test_min_samples_per_leaf_fraction_bounds_the_pieces_circuit_size() -> None:
    """
    Each floor piece carries near-unique ``id`` and ``source_id`` values, so
    fitting the "pieces" RSPN with the library's default ``min_samples_per_leaf``
    of one lets its circuit grow roughly one leaf per training piece --
    thousands of nodes even for a modest training set. Grounding deep-copies
    that circuit once per sampled piece, which exhausted memory before a room
    could be sampled. ``_MIN_SAMPLES_PER_LEAF_FRACTION`` must keep the fitted
    circuit small enough that grounding stays cheap.
    """
    rng = random.Random(0)
    layouts = [_synthetic_floor_layout(i, rng.randint(3, 8), rng) for i in range(60)]
    daos = [to_dao(layout) for layout in layouts]

    default_model = RelationalProbabilisticCircuit(EGRoomFloorLayout).fit(daos)
    bounded_model = RelationalProbabilisticCircuit(
        EGRoomFloorLayout, min_samples_per_leaf=_MIN_SAMPLES_PER_LEAF_FRACTION
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
