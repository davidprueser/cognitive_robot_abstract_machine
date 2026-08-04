from __future__ import annotations

from sqlalchemy.orm import Session

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from experiments.orm.ormatic_interface import (
    Base,
    Sage10kObjectDAO,
    Sage10kPositionDAO,
    Sage10kRotationDAO,
    Sage10kSizeDAO,
)
from experiments.sage_10k.populate_eg_objects_from_sage10k import (
    _eg_object_from_sage10k_object,
)
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.scene_generation.object_type_classifier import (
    ObjectTypeClassifier,
)
from semantic_digital_twin.scene_generation.scene_schema import ObjectType


def _make_sage10k_object(object_id: str, raw_type: str) -> Sage10kObjectDAO:
    return Sage10kObjectDAO(
        id=object_id,
        room_id="room_1",
        type=raw_type,
        description="a thing",
        source="generation",
        source_id=f"{object_id}_source",
        place_id="floor",
        place_guidance="on the floor",
        mass=1.0,
        position=Sage10kPositionDAO(x=1.0, y=2.0, z=0.5),
        rotation=Sage10kRotationDAO(x=0.0, y=0.0, z=45.0),
        dimensions=Sage10kSizeDAO(height=0.3, length=0.2, width=0.1),
    )


def test_eg_object_from_sage10k_object_maps_position_scale_and_orientation() -> None:
    """
    The converted EGObject must carry over id, room_id, place_id, source_id and
    every position/scale/orientation component unchanged.
    """
    sage10k_object = _make_sage10k_object("obj_1", "book2")

    eg_object = _eg_object_from_sage10k_object(sage10k_object, ObjectTypeClassifier())

    assert eg_object.id == "obj_1"
    assert eg_object.room_id == "room_1"
    assert eg_object.place_id == "floor"
    assert eg_object.source_id == "obj_1_source"
    assert eg_object.position.x == 1.0
    assert eg_object.position.y == 2.0
    assert eg_object.position.z == 0.5
    assert eg_object.orientation.x == 0.0
    assert eg_object.orientation.y == 0.0
    assert eg_object.orientation.z == 45.0
    assert eg_object.scale.height == 0.3
    assert eg_object.scale.length == 0.2
    assert eg_object.scale.width == 0.1


def test_eg_object_from_sage10k_object_maps_raw_type_to_generalized_object_type() -> None:
    """
    The raw, near-instance-specific ``type`` string must be resolved to a
    generalized ObjectType via the classifier, not stored verbatim.
    """
    sage10k_object = _make_sage10k_object("obj_2", "bookshelf")

    eg_object = _eg_object_from_sage10k_object(sage10k_object, ObjectTypeClassifier())

    assert eg_object.object_type == ObjectType.SHELF


def test_eg_object_from_sage10k_object_reads_from_a_persisted_row() -> None:
    """
    The conversion must work off a row round-tripped through a database session
    (eager-loaded relationships), not just a freshly constructed, still-
    transient dataclass instance.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = Session(engine)
    session.add(_make_sage10k_object("obj_3", "pottedplant"))
    session.commit()
    session.expire_all()

    sage10k_object = session.get(Sage10kObjectDAO, session.query(Sage10kObjectDAO).one().database_id)

    eg_object = _eg_object_from_sage10k_object(sage10k_object, ObjectTypeClassifier())

    assert eg_object.object_type == ObjectType.PLANT
    assert eg_object.id == "obj_3"
