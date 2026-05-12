import json
import os
from pathlib import Path
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from krrood.entity_query_language.factories import (
    variable,
    an,
    entity,
    contains,
    set_of,
    a,
)
from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.adapters.adaptive_environment_generation.sage10k_processing import (
    EGDataProcessing,
)
from semantic_digital_twin.adapters.adaptive_environment_generation.schema import (
    SceneGenerator,
    EGObject,
    EGRoom,
    EGPosition,
    EGSize,
    EGPoint2D,
    EGDoor,
    EGWall,
)
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.orm.ormatic_interface import *  # type: ignore
from semantic_digital_twin.world_description.geometry import Scale


def parse_json(path):
    json_directories = list(path.glob("*layout*"))
    json_files = []
    for json_directory in json_directories:
        [json_file] = list(json_directory.glob("layout_*.json"))
        json_files.append(json_file)
    results = []
    for file in json_files:
        raw_json = file.read_text()
        json_dict = json.loads(raw_json)
        result = SceneGenerator._from_json(json_dict)
        result.directory = path
        results.append(result)

    return results


def add_to_database(session):
    path = Path.home() / "Downloads" / "sage-10k-layouts"
    if not path.exists():
        raise ValueError(f"Path {path} does not exist")
    daos = []
    dao_state = ToDataAccessObjectState()
    for scene in parse_json(path)[0:100]:
        dao = to_dao(scene, dao_state)
        daos.append(dao)
        # print(dao)
        session.add(dao)
    session.commit()
    return session


def query_for_shelves(session):
    scenes = session.scalars(select(SceneGeneratorDAO)).all()
    objects = session.scalars(select(EGObjectDAO)).all()

    dao_state = FromDataAccessObjectState()
    var = variable(SceneGeneratorDAO, scenes)
    var2 = variable(EGObjectDAO, objects)

    query = (
        a(set_of(var.id, var2))
        .where(var2.object_type == "shelf")
        .where(var.room.id == var2.room_id)
        .distinct()
        .limit(5)
        .tolist()
    )
    return {r[var.id]: r[var2] for r in query}


def create_environment(scene_to_obj):
    downloa = EGDataProcessing()
    paths = {
        downloa.download_specific_scene(id): source_id
        for id, source_id in scene_to_obj.items()
    }
    scene_generator = SceneGenerator(
        id="scene_1",
        room=EGRoom(
            id="room_1",
            room_type="living_room",
            scale=EGSize(0, 1, 2),
            position=EGPosition(0, 0, 0),
            objects=list(paths.values()),
            walls=[
                EGWall(
                    id="wall_1",
                    start_point=EGPoint2D(0, 0),
                    end_point=EGPoint2D(4, 0),
                    height=8,
                    thickness=2,
                ),
                EGWall(
                    id="wall_2",
                    start_point=EGPoint2D(4, 4),
                    end_point=EGPoint2D(8, 4),
                    height=8,
                    thickness=2,
                ),
                EGWall(
                    id="wall_3",
                    start_point=EGPoint2D(8, 0),
                    end_point=EGPoint2D(8, 4),
                    height=8,
                    thickness=2,
                ),
                EGWall(
                    id="wall_4",
                    start_point=EGPoint2D(0, 4),
                    end_point=EGPoint2D(4, 4),
                    height=8,
                    thickness=2,
                ),
            ],
            doors=[
                EGDoor(
                    id="door_1",
                    wall_id="wall_1",
                    position_on_wall=45,
                    width=1,
                    height=2,
                    opens_inward=False,
                )
            ],
        ),
    )

    world = scene_generator.create_world()
    x = 0
    # new_objects = parse_json(obj.parent)

    # for q in query:
    #     print(q)


if __name__ == "__main__":

    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    session = Session(engine)
    Base.metadata.create_all(bind=engine)
    # session = add_to_database(session)
    result = query_for_shelves(session)
    create_environment(result)
