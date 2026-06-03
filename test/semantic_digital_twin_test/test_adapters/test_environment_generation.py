import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.dialects.oracle.dictionary import all_objects
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
from krrood.parametrization.model_registries import (
    ModelRegistry,
    DictRegistry,
    RelationalCircuitRegistry,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
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
    EGShelf,
    ObjectType,
    build_source_id_to_path,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.orm.ormatic_interface import *  # type: ignore
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world_description.geometry import Scale, BoundingBox


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


def query_environments(session):
    scenes = session.scalars(select(SceneGeneratorDAO)).all()
    objects = session.scalars(select(EGObjectDAO)).all()
    return scenes, objects


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
    return scene_generator, world


def test_simple_manual_environment(rclpy_node):
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    session = Session(engine)
    Base.metadata.create_all(bind=engine)

    # session = add_to_database(session)
    result = query_for_shelves(session)
    scene_generator, world = create_environment(result)
    viz_marker = VizMarkerPublisher(node=rclpy_node, _world=world)
    viz_marker.with_tf_publisher()


def test_simple_underspecified_environment(rclpy_node):
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    session = Session(engine)
    # Base.metadata.create_all(bind=engine)
    scene_generator, world = query_environments(session)
    rooms = [scene.room for scene in scene_generator]

    underspecified_scene_generator = underspecified(SceneGenerator)(
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
    # The RSPN models EGRoom directly: that is the class that owns the exchangeable parts
    # (objects, walls, doors) via ONETOMANY relationships in the DAO.
    underspecified_room = underspecified_scene_generator.kwargs["room"]
    rspn = RelationalProbabilisticCircuit(EGRoom)
    rspn = rspn.fit(rooms)

    registry = RelationalCircuitRegistry(
        relational_probabilistic_circuit=rspn, query=underspecified_room
    )
    prob_backend = ProbabilisticBackend(model_registry=registry, number_of_samples=1)
    samples = list(prob_backend.evaluate(underspecified_room))

    for room_sample in samples:
        scene = SceneGenerator(id="generated", room=room_sample)
        world = scene.create_world()

    viz_marker = VizMarkerPublisher(node=rclpy_node, _world=world)
    viz_marker.with_tf_publisher()


def _query_shelves_with_contents(session: Session) -> List[EGShelf]:
    """Load all scenes and group objects by the shelf they are placed on.

    Uses bounding-box spatial reasoning (not place_id alone) so the grouping
    mirrors what a robot would perceive from geometry rather than metadata.
    """
    dao_state = FromDataAccessObjectState()
    rooms = session.scalars(select(EGRoomDAO)).all()
    source_id_to_path = build_source_id_to_path()

    result: List[EGShelf] = []
    for room in rooms:
        room: EGRoom = room.from_dao(dao_state)

        shelves = [obj for obj in room.objects if obj.object_type == ObjectType.SHELF]
        non_shelves = [
            obj for obj in room.objects if obj.object_type != ObjectType.SHELF
        ]

        for shelf in shelves:
            region = BoundingBox(
                min_x=shelf.position.x - shelf.scale.width / 2,
                max_x=shelf.position.x + shelf.scale.width / 2,
                min_y=shelf.position.y - shelf.scale.length / 2,
                max_y=shelf.position.y + shelf.scale.length / 2,
                min_z=shelf.position.z - shelf.scale.height / 2,
                max_z=shelf.position.z + shelf.scale.height / 2,
                origin=HomogeneousTransformationMatrix(reference_frame=None),
            )
            contents = [
                non_shelf
                for non_shelf in non_shelves
                if region.contains(
                    Point3(
                        non_shelf.position.x,
                        non_shelf.position.y,
                        non_shelf.position.z,
                        None,
                    )
                )
            ]
            if not contents:
                continue
            result.append(
                EGShelf.create_shelf_with_contents(shelf, contents, source_id_to_path)
            )

    return result


def test_rspn_fitting_on_shelves(rclpy_node):
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    if not session.scalars(select(EGRoomDAO)).first():
        session = add_to_database(session)

    shelves = _query_shelves_with_contents(session)
    assert shelves, "No shelves with objects found — check the sage-10k-layouts path."

    shelf_daos = [to_dao(shelf) for shelf in shelves]

    rspn = RelationalProbabilisticCircuit(EGShelf)
    rspn.fit(shelf_daos)

    assert rspn.class_probabilistic_circuit is not None
    assert rspn.class_probabilistic_circuit.is_valid()
    assert "objects" in rspn.exchangeable_distribution_templates

    class_var_names = {v.name for v in rspn.class_probabilistic_circuit.variables}
    assert "EGShelf.position.x" in class_var_names
    assert "EGObjectOnShelfAggregations.total_object_count()" in class_var_names

    num_objects = 10
    query = underspecified(EGShelf)(
        position=underspecified(EGPosition)(x=..., y=..., z=...),
        scale=underspecified(EGSize)(width=..., length=..., height=...),
        orientation=underspecified(EGOrientation)(x=..., y=..., z=...),
        source_id=None,
        objects=[
            underspecified(EGObject)(
                id=None,
                room_id=None,
                place_id=None,
                object_type=...,
                scale=underspecified(EGSize)(width=..., length=..., height=...),
                position=underspecified(EGPosition)(x=..., y=..., z=...),
                orientation=underspecified(EGOrientation)(x=..., y=..., z=...),
                source_id=None,
            )
            for _ in range(num_objects)
        ],
    )

    registry = RelationalCircuitRegistry(
        relational_probabilistic_circuit=rspn, query=query
    )
    prob_backend = ProbabilisticBackend(model_registry=registry, number_of_samples=1)
    [shelf_sample] = list(prob_backend.evaluate(query))

    # Pick a random training shelf for the mesh; the RSPN provides geometry only.
    training_shelf = random.choice(
        [s for s in shelves if s.shelf_scene_dir is not None]
    )
    print(training_shelf.source_id)
    shelf_sample.source_id = training_shelf.source_id
    shelf_sample.shelf_scene_dir = training_shelf.shelf_scene_dir

    merged: Dict[ObjectType, list] = defaultdict(list)
    for shelf in shelves:
        if shelf.object_type_to_source_ids:
            for obj_type, entries in shelf.object_type_to_source_ids.items():
                merged[obj_type].extend(entries)
    shelf_sample.object_type_to_source_ids = merged

    world = shelf_sample.create_world()

    viz_marker = VizMarkerPublisher(node=rclpy_node, _world=world)
    viz_marker.with_tf_publisher()
