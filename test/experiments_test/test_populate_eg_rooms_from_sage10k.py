from __future__ import annotations

from sqlalchemy.orm import Session

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from experiments.orm.ormatic_interface import (
    Base,
    EGPositionDAO,
    EGRoomDAO,
    EGScaleDAO,
    Sage10kDoorDAO,
    Sage10kPositionDAO,
    Sage10kRoomDAO,
    Sage10kRoomDAO_doors_association,
    Sage10kRoomDAO_walls_association,
    Sage10kSizeDAO,
    Sage10kWallDAO,
    SceneGeneratorDAO,
)
from experiments.sage_10k.populate_eg_rooms_from_sage10k import (
    _delete_existing_eg_rooms,
    _eg_room_from_sage10k_room,
)
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.scene_generation.room_type_classifier import (
    RoomTypeClassifier,
)
from semantic_digital_twin.scene_generation.scene_schema import RoomType


def _make_sage10k_room(
    room_id: str, raw_room_type: str, width: float = 4.0, length: float = 5.0
) -> Sage10kRoomDAO:
    corners = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, length)),
        ((0.0, length), (width, length)),
        ((0.0, 0.0), (0.0, length)),
    ]
    return Sage10kRoomDAO(
        id=room_id,
        room_type=raw_room_type,
        floor_material="wood",
        dimensions=Sage10kSizeDAO(height=2.7, length=length, width=width),
        position=Sage10kPositionDAO(x=0.0, y=0.0, z=0.0),
        walls=[
            Sage10kRoomDAO_walls_association(
                target=Sage10kWallDAO(
                    id=f"{room_id}_wall_{index}",
                    start_point=Sage10kPositionDAO(x=start[0], y=start[1], z=0.0),
                    end_point=Sage10kPositionDAO(x=end[0], y=end[1], z=0.0),
                    material="plaster",
                    height=2.7,
                    thickness=0.1,
                )
            )
            for index, (start, end) in enumerate(corners)
        ],
        doors=[
            Sage10kRoomDAO_doors_association(
                target=Sage10kDoorDAO(
                    id=f"{room_id}_door_0",
                    wall_id=f"{room_id}_wall_0",
                    position_on_wall=0.5,
                    width=0.9,
                    height=2.0,
                    door_type="hinged",
                    opens_inward=True,
                    opening=False,
                    door_material="wood",
                )
            )
        ],
    )


def test_eg_room_from_sage10k_room_maps_raw_room_type_to_generalized_room_type() -> None:
    """
    The raw, inconsistently spelled ``room_type`` string must be resolved to a
    generalized RoomType via the classifier, not stored verbatim.
    """
    sage10k_room = _make_sage10k_room("room_1", "restaurant_dining_area")

    eg_room = _eg_room_from_sage10k_room(sage10k_room, RoomTypeClassifier())

    assert eg_room.room_type == RoomType.RESTAURANT


def test_eg_room_from_sage10k_room_takes_its_footprint_from_the_real_dimensions() -> None:
    """
    The room footprint must come from the dataset's own dimensions rather than
    being derived from the objects standing in it.
    """
    sage10k_room = _make_sage10k_room("room_2", "kitchen", width=4.2, length=3.8)

    eg_room = _eg_room_from_sage10k_room(sage10k_room, RoomTypeClassifier())

    assert eg_room.id == "room_2"
    assert eg_room.scale.width == 4.2
    assert eg_room.scale.length == 3.8
    assert eg_room.scale.height == 2.7


def test_eg_room_from_sage10k_room_carries_over_walls_and_doors() -> None:
    """
    Walls and doors are what make a generated room enclosable, so both must be
    carried across with their geometry intact.
    """
    sage10k_room = _make_sage10k_room("room_3", "office", width=4.0, length=5.0)

    eg_room = _eg_room_from_sage10k_room(sage10k_room, RoomTypeClassifier())

    assert len(eg_room.walls) == 4
    assert {(wall.start_point.x, wall.start_point.y) for wall in eg_room.walls} == {
        (0.0, 0.0),
        (4.0, 0.0),
        (0.0, 5.0),
    }
    assert all(wall.height == 2.7 and wall.thickness == 0.1 for wall in eg_room.walls)
    assert len(eg_room.doors) == 1
    assert eg_room.doors[0].wall_id == "room_3_wall_0"
    assert eg_room.doors[0].width == 0.9
    assert eg_room.doors[0].opens_inward is True


def test_eg_room_from_sage10k_room_leaves_the_objects_association_empty() -> None:
    """
    Floor pieces are joined on ``EGObject.room_id`` at extraction time, so
    populating the association as well would duplicate a quarter-million rows.
    """
    sage10k_room = _make_sage10k_room("room_4", "kitchen")

    eg_room = _eg_room_from_sage10k_room(sage10k_room, RoomTypeClassifier())

    assert eg_room.objects == []


def test_delete_existing_eg_rooms_also_removes_the_scene_generators_holding_them() -> None:
    """
    ``SceneGeneratorDAO.room_id`` is a foreign key onto ``EGRoomDAO``, so a
    re-run that deletes only the rooms hits a foreign key violation and leaves
    the table unchanged.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = Session(engine)
    room = EGRoomDAO(
        id="stale_room",
        room_type=RoomType.LIVING_ROOM,
        scale=EGScaleDAO(height=2.7, length=5.0, width=4.0),
        position=EGPositionDAO(x=0.0, y=0.0, z=0.0),
    )
    session.add(SceneGeneratorDAO(id="stale_scene", room=room))
    session.commit()

    deleted_count = _delete_existing_eg_rooms(session)

    assert deleted_count == 1
    assert session.query(EGRoomDAO).count() == 0
    assert session.query(SceneGeneratorDAO).count() == 0


def test_eg_room_from_sage10k_room_reads_from_a_persisted_row() -> None:
    """
    The conversion must work off a row round-tripped through a database session
    (eager-loaded relationships), not just a freshly constructed, still-
    transient dataclass instance.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = Session(engine)
    session.add(_make_sage10k_room("room_5", "master bedroom"))
    session.commit()
    session.expire_all()

    sage10k_room = session.query(Sage10kRoomDAO).one()

    eg_room = _eg_room_from_sage10k_room(sage10k_room, RoomTypeClassifier())

    assert eg_room.room_type == RoomType.BEDROOM
    assert eg_room.id == "room_5"
    assert len(eg_room.walls) == 4
