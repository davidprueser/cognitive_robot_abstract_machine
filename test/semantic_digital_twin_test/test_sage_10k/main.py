import json
import os
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from krrood.entity_query_language.factories import variable, an, entity, contains
from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.adapters.adaptive_environment_generation.schema import (
    SceneGenerator,
    EGObject,
)
from semantic_digital_twin.orm.ormatic_interface import *  # type: ignore


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

    results = an(entity(var2)).where(var2.object_type == "shelf").evaluate()
    new_var = variable(EGObjectDAO, list(results))

    query = an(entity(var)).where(var.room_id == new_var.room_id).limit(5)
    result = query.evaluate()
    print(*result, sep="\n")

    #
    # for q in query:
    #     print(q)


if __name__ == "__main__":

    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    session = Session(engine)
    Base.metadata.create_all(bind=engine)
    session = add_to_database(session)
    query_for_shelves(session)
