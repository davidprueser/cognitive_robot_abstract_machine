from __future__ import annotations

import dataclasses
import os
from collections.abc import Collection

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload
from typing_extensions import TYPE_CHECKING

from krrood.ormatic.utils import create_engine
from semantic_digital_twin.scene_generation.scene_schema import (
    EGShelf,
    EGShelfLayer,
    ObjectType,
)

if TYPE_CHECKING:
    from experiments.orm.ormatic_interface import EGObjectDAO


def load_objects_of_types(
    session: Session, object_types: Collection[ObjectType]
) -> list[EGObjectDAO]:
    """
    Load every object DAO whose type is one of *object_types*, eagerly joining
    scale/position/orientation, for use as a mesh-candidate pool.

    Restricting the query to the types a caller actually needs -- rather than pulling
    every object regardless of type -- keeps a mesh-candidate lookup proportional to
    what a sampled shelf or object can use instead of the whole database.

    Deliberately independent of :func:`load_shelf_layers`: which layers an RSPN is
    trained on must not also narrow which meshes are available to dress the sampled
    result, so callers building a mesh-candidate pool should use this instead of the
    objects a training extractor happened to load.

    :param session: Database session to query objects from.
    :param object_types: Only objects whose type is one of these are included.
    :return: Loaded object DAOs of the given types.
    """
    from experiments.orm.ormatic_interface import EGObjectDAO

    return session.scalars(
        select(EGObjectDAO)
        .where(EGObjectDAO.object_type.in_(object_types))
        .options(
            joinedload(EGObjectDAO.scale),
            joinedload(EGObjectDAO.position),
            joinedload(EGObjectDAO.orientation),
        )
        .distinct()
    ).all()


def load_objects_with_cached_meshes(
    session: Session, cached_source_ids: Collection[str]
) -> list[EGObjectDAO]:
    """
    Load every object DAO whose mesh is already cached locally, eagerly joining
    scale/position/orientation.

    Selecting by mesh availability keeps the candidate pool complete and reproducible.
    Capping an unordered query first and intersecting with the cached meshes afterwards
    left the pool an accident of which rows the database happened to return -- a few
    dozen candidates skewed towards whichever types earlier demos had downloaded, so
    most sampled object types found no mesh of their own kind and fell back to the whole
    pool. The result is bounded by the size of the local cache, so it needs no row
    limit.

    :param session: Database session to query objects from.
    :param cached_source_ids: Source IDs whose mesh files are cached locally.
    :return: All object DAOs whose mesh is available.
    """
    from experiments.orm.ormatic_interface import EGObjectDAO

    return session.scalars(
        select(EGObjectDAO)
        .where(EGObjectDAO.source_id.in_(cached_source_ids))
        .options(
            joinedload(EGObjectDAO.scale),
            joinedload(EGObjectDAO.position),
            joinedload(EGObjectDAO.orientation),
        )
        .distinct()
    ).all()


def load_shelf_layers(
    session: Session, object_type: ObjectType | None = None
) -> list[EGShelfLayer]:
    """
    Load every shelf layer prepared by the preprocessing pipeline.

    The stored layers already carry mesh-centred positions, unified object
    types and content-frame poses, so fitting a circuit on them needs no
    further processing -- which is what keeps mesh measurement and clustering
    out of every training run.

    Every relationship reached while converting a layer is eagerly loaded.
    Leaving the objects' own scale, position and orientation to lazy loading
    costs three statements per object on the one query path every training run
    takes -- which is the very cost preprocessing exists to remove.

    :param session: Session on the processed database.
    :param object_type: When given, layers are reduced to their objects of this
        type and layers left empty are dropped.
    :return: The stored shelf layers.
    """
    from experiments.orm.ormatic_interface import (
        EGObject2DDAO,
        EGShelfLayerDAO,
        EGShelfLayerDAO_objects_association,
    )

    layer_data_access_objects = (
        session.scalars(
            select(EGShelfLayerDAO).options(
                joinedload(EGShelfLayerDAO.objects)
                .joinedload(EGShelfLayerDAO_objects_association.target)
                .options(
                    joinedload(EGObject2DDAO.scale),
                    joinedload(EGObject2DDAO.position),
                    joinedload(EGObject2DDAO.orientation),
                ),
            )
        )
        .unique()
        .all()
    )
    layers = [
        layer_data_access_object.from_dao()
        for layer_data_access_object in layer_data_access_objects
    ]
    if object_type is None:
        return layers

    matching_layers = [
        dataclasses.replace(
            layer,
            objects=[obj for obj in layer.objects if obj.object_type == object_type],
        )
        for layer in layers
    ]
    return [layer for layer in matching_layers if layer.objects]


def load_shelves(session: Session) -> list[EGShelf]:
    """
    Load every shelf prepared by the preprocessing pipeline, with its layers and their
    objects.

    Loading whole shelves rather than loose layers is what lets a circuit learn
    how many layers a kind of shelf has and how they are spaced, alongside what
    stands on them.

    The join chain is extended one level past :func:`load_shelf_layers` for the
    same reason it exists there: leaving a level to lazy loading costs a
    statement per row on the one query path every training run takes.

    :param session: Session on the processed database.
    :return: The stored shelves.
    """
    from experiments.orm.ormatic_interface import (
        EGObject2DDAO,
        EGShelfDAO,
        EGShelfDAO_layers_association,
        EGShelfLayerDAO,
        EGShelfLayerDAO_objects_association,
    )

    shelf_data_access_objects = (
        session.scalars(
            select(EGShelfDAO).options(
                joinedload(EGShelfDAO.scale),
                joinedload(EGShelfDAO.layers)
                .joinedload(EGShelfDAO_layers_association.target)
                .options(
                    joinedload(EGShelfLayerDAO.objects)
                    .joinedload(EGShelfLayerDAO_objects_association.target)
                    .options(
                        joinedload(EGObject2DDAO.scale),
                        joinedload(EGObject2DDAO.position),
                        joinedload(EGObject2DDAO.orientation),
                    ),
                ),
            )
        )
        .unique()
        .all()
    )
    return [
        shelf_data_access_object.from_dao()
        for shelf_data_access_object in shelf_data_access_objects
    ]


def _processed_database_session() -> Session:
    """
    Open a session on the processed sage10k database, creating its schema if needed.

    :return: A session on the processed database.
    """
    from experiments.orm.ormatic_interface import Base

    uri = os.environ.get("SAGE10K_PROCESSED_DATABASE_URI")
    assert (
        uri is not None
    ), "Please set the SAGE10K_PROCESSED_DATABASE_URI environment variable."
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    return Session(engine)
