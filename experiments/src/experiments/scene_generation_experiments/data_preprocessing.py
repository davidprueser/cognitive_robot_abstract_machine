from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from experiments.orm.ormatic_interface import (
    Base,
    Sage10kObjectDAO,
    Sage10kRoomDAO,
    Sage10kSceneDAO,
    Sage10kSceneDAO_rooms_association,
)
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.scene_generation.sage10k_processing import EGDataProcessing
from semantic_digital_twin.semantic_annotations.description_matching import DescriptionCategoryScorer


class SourceIdNotFoundError(Exception):
    """
    Raised when no Sage-10k object matching a given ``source_id`` can be found
    in the database.
    """

    def __init__(self, source_id: str) -> None:
        super().__init__(
            f"No Sage-10k object with source_id '{source_id}' found in the database."
        )
        self.source_id = source_id
        """
        The ``source_id`` that could not be resolved.
        """


@dataclass
class Sage10kSceneDownloader:
    """
    Resolves a Sage-10k object's ``source_id`` to the scene it belongs to and
    downloads that scene from HuggingFace.

    After a successful download the PLY mesh is located at
    ``<returned_path>/objects/<source_id>.ply`` and the texture at
    ``<returned_path>/objects/<source_id>_texture.png``.
    """

    session: Session
    """
    SQLAlchemy session used to look up scene membership from the Sage-10k
    database.
    """

    data_processor: EGDataProcessing = field(default_factory=EGDataProcessing)
    """
    Handles local caching and downloading of Sage-10k scene archives from
    HuggingFace.
    """

    def download_scene_for_source_id(self, source_id: str) -> Path:
        """
        Download and return the scene directory containing the object
        identified by *source_id*.

        :param source_id: The ``source_id`` of a Sage-10k object as
            stored in the database.
        :return: Path to the downloaded and extracted scene directory.
        :raises SourceIdNotFoundError: If *source_id* does not match any
            object in the database.
        """
        layout_name = self._find_layout_name(source_id)
        return self.data_processor.download_specific_scene(layout_name)

    def _find_layout_name(self, source_id: str) -> str:
        """
        Traverse the database from the object identified by *source_id* up to
        its containing scene and return that scene's layout name.

        :param source_id: The source ID to resolve.
        :return: The layout name (``id`` field) of the containing scene,
            e.g. ``"fd6894a7"``.
        :raises SourceIdNotFoundError: If the traversal fails at any
            step.
        """
        object_record = self.session.scalars(
            select(Sage10kObjectDAO)
            .where(Sage10kObjectDAO.source_id == source_id)
            .limit(1)
        ).first()
        if object_record is None:
            raise SourceIdNotFoundError(source_id)

        room = self.session.scalars(
            select(Sage10kRoomDAO)
            .where(Sage10kRoomDAO.id == object_record.room_id)
            .limit(1)
        ).first()
        if room is None:
            raise SourceIdNotFoundError(source_id)

        association = self.session.scalars(
            select(Sage10kSceneDAO_rooms_association).where(
                Sage10kSceneDAO_rooms_association.target_sage10kroomdao_id
                == room.database_id
            )
        ).first()
        if association is None:
            raise SourceIdNotFoundError(source_id)

        scene = self.session.get(Sage10kSceneDAO, association.source_sage10kscenedao_id)
        if scene is None:
            raise SourceIdNotFoundError(source_id)

        return scene.id


def demo() -> None:
    """
    Print NLP description category scores for the first 1000 Sage-10k objects
    found in the configured database.
    """
    uri = os.environ.get("SAGE10k_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    objects = session.scalars(select(Sage10kObjectDAO).limit(1000)).all()
    for sage_object in objects:
        scorer = DescriptionCategoryScorer()
        score = scorer.score("books", sage_object.description).score
        print(sage_object.description)
        print(score)


if __name__ == "__main__":
    demo()
