import json
import os
from pathlib import Path
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import rclpy_node
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import (
    variable,
    an,
    entity,
    contains,
    set_of,
    a,
    underspecified,
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
    EGOrientation,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.orm.ormatic_interface import *  # type: ignore
from semantic_digital_twin.reasoning.predicates import InsideOf
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
    objects = {
        downloa.download_specific_scene(id): source_id
        for id, source_id in scene_to_obj.items()
    }
    objects = {directory: obj.from_dao() for directory, obj in objects.items()}
    scene_generator = SceneGenerator(
        id="scene_1",
        mesh_to_object_mapping=objects,
        room=EGRoom(
            id="room_1",
            room_type="living_room",
            scale=EGSize(0, 1, 2),
            position=EGPosition(0, 0, 0),
            objects=list(objects.values()),
            walls=[
                EGWall(
                    id="wall_1",
                    start_point=EGPoint2D(0.0, 5.0),
                    end_point=EGPoint2D(5.5, 5.0),
                    height=2.7,
                    thickness=0.1,
                ),
                EGWall(
                    id="wall_2",
                    start_point=EGPoint2D(0, 0),
                    end_point=EGPoint2D(5.5, 0),
                    height=2.7,
                    thickness=0.1,
                ),
                EGWall(
                    id="wall_3",
                    start_point=EGPoint2D(5.5, 0),
                    end_point=EGPoint2D(5.5, 5.0),
                    height=2.7,
                    thickness=0.1,
                ),
                EGWall(
                    id="wall_4",
                    start_point=EGPoint2D(0, 0),
                    end_point=EGPoint2D(0, 5.0),
                    height=2.7,
                    thickness=0.1,
                ),
            ],
            doors=[
                EGDoor(
                    id="door_1",
                    wall_id="wall_1",
                    position_on_wall=0.42,
                    width=0.95,
                    height=2.05,
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
    return world


def test_simple_manual_environment(rclpy_node):
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    session = Session(engine)
    Base.metadata.create_all(bind=engine)

    # session = add_to_database(session)
    result = query_for_shelves(session)
    world = create_environment(result)
    viz_marker = VizMarkerPublisher(node=rclpy_node, _world=world)
    viz_marker.with_tf_publisher()


def test_simple_underspecified_environment(rclpy_node):
    # uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    # engine = create_engine(uri)
    # session = Session(engine)
    # Base.metadata.create_all(bind=engine)

    scene_generator = underspecified(SceneGenerator)(
        id=None,
        mesh_to_object_mapping=None,
        room=underspecified(EGRoom)(
            id=None,
            room_type=None,
            scale=underspecified(EGSize)(width=..., length=..., height=...),
            position=underspecified(EGPosition)(x=..., y=..., z=...),
            objects=[
                underspecified(EGObject)(
                    id=None,
                    room_id=None,
                    place_id=None,
                    object_type=None,
                    scale=underspecified(EGSize)(width=..., length=..., height=...),
                    position=underspecified(EGPosition)(x=..., y=..., z=...),
                    orientation=underspecified(EGOrientation)(x=..., y=..., z=...),
                    source_id=None,
                ),
                underspecified(EGObject)(
                    id=None,
                    room_id=None,
                    place_id=None,
                    object_type=None,
                    scale=underspecified(EGSize)(width=..., length=..., height=...),
                    position=underspecified(EGPosition)(x=..., y=..., z=...),
                    orientation=underspecified(EGOrientation)(x=..., y=..., z=...),
                    source_id=None,
                ),
            ],
            walls=[
                EGWall(
                    id="wall_1",
                    start_point=EGPoint2D(0.0, 5.0),
                    end_point=EGPoint2D(5.5, 5.0),
                    height=2.7,
                    thickness=0.1,
                ),
                EGWall(
                    id="wall_2",
                    start_point=EGPoint2D(0, 0),
                    end_point=EGPoint2D(5.5, 0),
                    height=2.7,
                    thickness=0.1,
                ),
                EGWall(
                    id="wall_3",
                    start_point=EGPoint2D(5.5, 0),
                    end_point=EGPoint2D(5.5, 5.0),
                    height=2.7,
                    thickness=0.1,
                ),
                EGWall(
                    id="wall_4",
                    start_point=EGPoint2D(0, 0),
                    end_point=EGPoint2D(0, 5.0),
                    height=2.7,
                    thickness=0.1,
                ),
            ],
            doors=[
                EGDoor(
                    id="door_1",
                    wall_id="wall_1",
                    position_on_wall=0.42,
                    width=0.95,
                    height=2.05,
                    opens_inward=False,
                ),
            ],
        ),
    )
    scene_generator.resolve()
    scene_generator.where(
        InsideOf(scene_generator.variable.objects[0], scene_generator.variable.room)
    )

    prob_backend = ProbabilisticBackend(number_of_samples=1)
    samples = prob_backend.evaluate(scene_generator)
    samples = list(samples)

    for sample in samples:
        world = sample.create_world()

    viz_marker = VizMarkerPublisher(node=rclpy_node, _world=world)
    viz_marker.with_tf_publisher()
