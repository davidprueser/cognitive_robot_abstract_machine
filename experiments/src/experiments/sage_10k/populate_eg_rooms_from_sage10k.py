from __future__ import annotations

import os
import time

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, joinedload

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.orm.ormatic_interface import *  # type: ignore
from semantic_digital_twin.scene_generation.room_type_classifier import (
    RoomTypeClassifier,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGDoor,
    EGPoint2D,
    EGPosition,
    EGRoom,
    EGScale,
    EGWall,
)

COMMIT_BATCH_SIZE = 500


def _eg_wall_from_sage10k_wall(sage10k_wall: Sage10kWallDAO) -> EGWall:
    """
    Build an :class:`EGWall` equivalent of *sage10k_wall*, keeping only the
    two-dimensional endpoints that define the room's footprint.

    :param sage10k_wall: Raw wall row loaded from the sage10k database.
    :return: The equivalent :class:`EGWall`.
    """
    return EGWall(
        id=sage10k_wall.id,
        start_point=EGPoint2D(
            x=sage10k_wall.start_point.x, y=sage10k_wall.start_point.y
        ),
        end_point=EGPoint2D(x=sage10k_wall.end_point.x, y=sage10k_wall.end_point.y),
        height=sage10k_wall.height,
        thickness=sage10k_wall.thickness,
    )


def _eg_door_from_sage10k_door(sage10k_door: Sage10kDoorDAO) -> EGDoor:
    """
    Build an :class:`EGDoor` equivalent of *sage10k_door*.

    :param sage10k_door: Raw door row loaded from the sage10k database.
    :return: The equivalent :class:`EGDoor`.
    """
    return EGDoor(
        id=sage10k_door.id,
        wall_id=sage10k_door.wall_id,
        position_on_wall=sage10k_door.position_on_wall,
        width=sage10k_door.width,
        height=sage10k_door.height,
        opens_inward=sage10k_door.opens_inward,
    )


def _eg_room_from_sage10k_room(
    sage10k_room: Sage10kRoomDAO, classifier: RoomTypeClassifier
) -> EGRoom:
    """
    Build an :class:`EGRoom` equivalent of *sage10k_room*, mapping its
    free-form ``room_type`` string onto a generalized :class:`RoomType` via
    *classifier*.

    The ``objects`` association is deliberately left empty: floor pieces are
    joined on :attr:`EGObject.room_id` when layouts are extracted, so filling it
    here would duplicate a quarter-million rows for no gain.

    :param sage10k_room: Raw room row loaded from the sage10k database, with its
        ``dimensions``, ``position``, ``walls`` and ``doors`` relationships
        already loaded.
    :param classifier: Maps *sage10k_room*'s raw room type string onto a
        :class:`RoomType`.
    :return: The equivalent :class:`EGRoom`.
    """
    return EGRoom(
        id=sage10k_room.id,
        room_type=classifier.classify(sage10k_room.room_type),
        scale=EGScale(
            height=sage10k_room.dimensions.height,
            length=sage10k_room.dimensions.length,
            width=sage10k_room.dimensions.width,
        ),
        position=EGPosition(
            x=sage10k_room.position.x,
            y=sage10k_room.position.y,
            z=sage10k_room.position.z,
        ),
        walls=[
            _eg_wall_from_sage10k_wall(association.target)
            for association in sage10k_room.walls
        ],
        doors=[
            _eg_door_from_sage10k_door(association.target)
            for association in sage10k_room.doors
        ],
    )


def _delete_existing_eg_rooms(session: Session) -> int:
    """
    Remove every stored :class:`EGRoom`, its association rows and the
    :class:`SceneGenerator` rows holding it, so a re-run replaces the table
    rather than appending a second copy of the dataset.

    ``SceneGeneratorDAO.room_id`` is a foreign key onto ``EGRoomDAO``, so those
    rows have to go first or the delete fails outright.

    .. warning::
        The ORM mapping defines no cascade from a room onto the wall, door,
        point and scale rows it references, so each re-run leaves those behind
        as unreferenced rows. They are never loaded again -- every read reaches
        them through a room association -- but the tables do grow by roughly
        50k rows per run.

    :param session: Session on the semantic_digital_twin database.
    :return: Number of deleted room rows.
    """
    session.execute(delete(SceneGeneratorDAO))
    for association in (
        EGRoomDAO_objects_association,
        EGRoomDAO_walls_association,
        EGRoomDAO_doors_association,
        EGRoomDAO_shelves_association,
        EGRoomDAO_groups_association,
    ):
        session.execute(delete(association))
    deleted_count = session.query(EGRoomDAO).delete()
    session.commit()
    return deleted_count


def populate_eg_rooms_from_sage10k() -> None:
    """
    Read every Sage10k room from the sage10k database and store an equivalent
    :class:`EGRoom` -- carrying its real footprint, walls, doors and a
    generalized :class:`RoomType` -- in the semantic_digital_twin database.

    Room generation previously fabricated each room's footprint from the
    bounding box of the pieces standing in it, which produced rooms far smaller
    than the real ones. These rows are what let it use the dataset's own
    geometry instead.

    Rooms are committed in batches to keep memory usage bounded across the full
    dataset.
    """
    sage10k_database_uri = os.environ.get("SAGE10k_DATABASE_URI")
    semantic_digital_twin_database_uri = os.environ.get(
        "SEMANTIC_DIGITAL_TWIN_DATABASE_URI"
    )
    assert (
        sage10k_database_uri is not None
    ), "Please set the SAGE10k_DATABASE_URI environment variable."
    assert (
        semantic_digital_twin_database_uri is not None
    ), "Please set the SEMANTIC_DIGITAL_TWIN_DATABASE_URI environment variable."

    sage10k_engine = create_engine(sage10k_database_uri)
    sage10k_session = Session(sage10k_engine)

    semantic_digital_twin_engine = create_engine(semantic_digital_twin_database_uri)
    Base.metadata.create_all(bind=semantic_digital_twin_engine)
    semantic_digital_twin_session = Session(semantic_digital_twin_engine)

    classifier = RoomTypeClassifier()

    deleted_count = _delete_existing_eg_rooms(semantic_digital_twin_session)
    print(f"Removed {deleted_count} previously stored EGRoom rows.")

    sage10k_rooms = sage10k_session.scalars(
        select(Sage10kRoomDAO).options(
            joinedload(Sage10kRoomDAO.dimensions),
            joinedload(Sage10kRoomDAO.position),
        )
    ).all()

    total = len(sage10k_rooms)
    print(f"Found {total} Sage10k rooms. Converting to EGRoom...")

    start = time.time()
    for index, sage10k_room in enumerate(sage10k_rooms):
        eg_room = _eg_room_from_sage10k_room(sage10k_room, classifier)
        semantic_digital_twin_session.add(to_dao(eg_room))
        if (index + 1) % COMMIT_BATCH_SIZE == 0:
            semantic_digital_twin_session.commit()
            semantic_digital_twin_session.expire_all()
            print(f"  committed {index + 1}/{total}")

    semantic_digital_twin_session.commit()
    print(f"Done. Converted {total} rooms in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    populate_eg_rooms_from_sage10k()
