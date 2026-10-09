from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from experiments.shelf_generation_experiments.dataset_environment import (
    DatasetEnvironmentVariable,
)
from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentShelf,
)
from experiments.shelf_generation_experiments.utils import (
    MeshCandidate,
    ObjectType,
    build_source_id_to_path,
)
from krrood.ormatic.utils import create_engine


# %% processed database
@dataclass
class ProcessedShelfDatabase:
    """
    Reads the shelves and objects that preprocessing stored.
    """

    session: Session
    """
    Session on the processed database.
    """

    @classmethod
    def from_environment(cls) -> ProcessedShelfDatabase:
        """
        :return: The processed database the environment names, with its schema created
            if it does not exist yet.
        """
        from experiments.orm.ormatic_interface import Base

        engine = create_engine(DatasetEnvironmentVariable.PROCESSED_DATABASE_URI.read())
        Base.metadata.create_all(bind=engine)
        return cls(session=Session(engine))

    def shelves(self) -> list[RelationalCircuitExperimentShelf]:
        """
        :return: Every stored shelf, with its layers and their objects.
        """
        from experiments.orm.ormatic_interface import (
            RelationalCircuitExperimentObject2DDAO,
            RelationalCircuitExperimentShelfDAO,
            RelationalCircuitExperimentShelfDAO_layers_association,
            RelationalCircuitExperimentShelfLayerDAO,
            RelationalCircuitExperimentShelfLayerDAO_objects_association,
        )

        shelf_data_access_objects = (
            self.session.scalars(
                select(RelationalCircuitExperimentShelfDAO).options(
                    joinedload(RelationalCircuitExperimentShelfDAO.scale),
                    joinedload(RelationalCircuitExperimentShelfDAO.layers)
                    .joinedload(
                        RelationalCircuitExperimentShelfDAO_layers_association.target
                    )
                    .options(
                        joinedload(RelationalCircuitExperimentShelfLayerDAO.objects)
                        .joinedload(
                            RelationalCircuitExperimentShelfLayerDAO_objects_association.target
                        )
                        .options(
                            joinedload(RelationalCircuitExperimentObject2DDAO.scale),
                            joinedload(RelationalCircuitExperimentObject2DDAO.pose),
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

    def mesh_candidates(
        self, object_types: Collection[ObjectType], scenes_root: Path
    ) -> list[MeshCandidate]:
        """
        :param object_types: The object types to collect meshes of.
        :param scenes_root: Directory holding the downloaded scenes whose meshes can be
            spawned.
        :return: One candidate per stored object of one of *object_types* whose mesh is
            available under *scenes_root*.
        """
        from experiments.orm.ormatic_interface import PreprocessedObjectDAO

        scene_directory_by_source_id = build_source_id_to_path(scenes_root)
        object_data_access_objects = self.session.scalars(
            select(PreprocessedObjectDAO)
            .where(PreprocessedObjectDAO.object_type.in_(object_types))
            .where(PreprocessedObjectDAO.source_id.in_(scene_directory_by_source_id))
            .options(joinedload(PreprocessedObjectDAO.scale))
        ).all()
        return [
            MeshCandidate(
                scene_directory=scene_directory_by_source_id[
                    object_data_access_object.source_id
                ],
                source_id=object_data_access_object.source_id,
                object_type=object_data_access_object.object_type,
                scale=object_data_access_object.scale.from_dao(),
            )
            for object_data_access_object in object_data_access_objects
        ]
