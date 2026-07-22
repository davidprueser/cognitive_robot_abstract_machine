from __future__ import annotations

import os
import time

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.orm.ormatic_interface import *  # type: ignore
from semantic_digital_twin.scene_generation.object_type_classifier import (
    ObjectTypeClassifier,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject,
    EGPosition,
    EGRotation,
    EGScale,
)

COMMIT_BATCH_SIZE = 500


def _eg_object_from_sage10k_object(
    sage10k_object: Sage10kObjectDAO, classifier: ObjectTypeClassifier
) -> EGObject:
    """
    Build an :class:`EGObject` equivalent of *sage10k_object*, mapping its
    free-form ``type`` string onto a generalized :class:`ObjectType` via
    *classifier*.

    :param sage10k_object: Raw object row loaded from the sage10k
        database, with its ``position``, ``rotation`` and ``dimensions``
        relationships already loaded.
    :param classifier: Maps *sage10k_object*'s raw type string onto an
        :class:`ObjectType`.
    :return: The equivalent :class:`EGObject`.
    """
    return EGObject(
        id=sage10k_object.id,
        room_id=sage10k_object.room_id,
        place_id=sage10k_object.place_id,
        object_type=classifier.classify(sage10k_object.type),
        scale=EGScale(
            height=sage10k_object.dimensions.height,
            length=sage10k_object.dimensions.length,
            width=sage10k_object.dimensions.width,
        ),
        position=EGPosition(
            x=sage10k_object.position.x,
            y=sage10k_object.position.y,
            z=sage10k_object.position.z,
        ),
        orientation=EGRotation(
            x=sage10k_object.rotation.x,
            y=sage10k_object.rotation.y,
            z=sage10k_object.rotation.z,
        ),
        source_id=sage10k_object.source_id,
    )


def populate_eg_objects_from_sage10k() -> None:
    """
    Read every Sage10k object from the sage10k database and store an equivalent
    :class:`EGObject` -- with its raw ``type`` string mapped to a generalized
    :class:`ObjectType` -- in the semantic_digital_twin database.

    Objects are committed in batches to keep memory usage bounded across
    the full dataset.
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

    classifier = ObjectTypeClassifier()

    sage10k_objects = sage10k_session.scalars(
        select(Sage10kObjectDAO).options(
            joinedload(Sage10kObjectDAO.position),
            joinedload(Sage10kObjectDAO.rotation),
            joinedload(Sage10kObjectDAO.dimensions),
        )
    ).all()

    total = len(sage10k_objects)
    print(f"Found {total} Sage10k objects. Converting to EGObject...")

    start = time.time()
    for index, sage10k_object in enumerate(sage10k_objects):
        eg_object = _eg_object_from_sage10k_object(sage10k_object, classifier)
        semantic_digital_twin_session.add(to_dao(eg_object))
        if (index + 1) % COMMIT_BATCH_SIZE == 0:
            semantic_digital_twin_session.commit()
            semantic_digital_twin_session.expire_all()
            print(f"  committed {index + 1}/{total}")

    semantic_digital_twin_session.commit()
    print(f"Done. Converted {total} objects in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    populate_eg_objects_from_sage10k()
