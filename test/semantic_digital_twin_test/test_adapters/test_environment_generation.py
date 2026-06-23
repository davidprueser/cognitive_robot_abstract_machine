import json
import os
import random
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock

import pytest

import numpy as np
import rclpy
from sklearn.cluster import DBSCAN
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
from semantic_digital_twin.adapters.adaptive_environment_generation.collision_resolution import (
    resolve_shelf_collisions,
)
from semantic_digital_twin.adapters.adaptive_environment_generation.schema import (
    SceneGenerator,
    EGObject,
    EGObject2D,
    EGShelfLayer,
    EGRoom,
    EGPosition,
    EGSize,
    EGPoint2D,
    EGDoor,
    EGWall,
    EGOrientation,
    EGShelf,
    ObjectType,
    BookObjectType,
    build_source_id_to_path,
)
from physics_simulators.base_simulator import SimulatorConstraints
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.adapters.partnet_mobility_dataset.loader import (
    PartNetMobilityDatasetLoader,
)
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.orm.ormatic_interface import *  # type: ignore
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.semantic_annotations.semantic_annotations import ShelfLayer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Scale, BoundingBox
from semantic_digital_twin.world_description.world_entity import Body


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


def _extract_shelf_layers_from_objects(
    session: Session, edge_margin_fraction: float = 0.10
) -> List[EGShelfLayer]:
    """
    Load all scenes and group objects by the shelf they are placed on.

    Uses bounding-box spatial reasoning (not place_id alone) so the grouping
    mirrors what a robot would perceive from geometry rather than metadata.

    :param edge_margin_fraction: Fraction of each shelf dimension to use as an
        inset margin on X and Y.  Objects whose centre falls within this margin
        of the shelf boundary are excluded from training data so the learned RSPN
        does not place books at positions where they would fall off in simulation.
    """
    dao_state = FromDataAccessObjectState()
    objects = session.scalars(select(EGObjectDAO).distinct().limit(50000)).all()
    shelf_layers = []

    shelves = [obj for obj in objects if obj.object_type == ObjectType.SHELF]
    non_shelves = [obj for obj in objects if BookObjectType.contains(obj.object_type)]

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
        inner_region = region.bloat(
            x_amount=-(edge_margin_fraction * shelf.scale.width),
            y_amount=-(edge_margin_fraction * shelf.scale.length),
            z_amount=0.0,
        )
        objects_in_shelf = [
            non_shelf
            for non_shelf in non_shelves
            if inner_region.contains(
                Point3(
                    non_shelf.position.x,
                    non_shelf.position.y,
                    non_shelf.position.z,
                    None,
                )
            )
        ]
        if not objects_in_shelf:
            continue

        position_z = [pos.position.z for pos in objects_in_shelf]
        z_to_np = np.array(position_z).reshape(-1, 1)

        labels = DBSCAN(
            eps=0.05,
            min_samples=1,
        ).fit_predict(z_to_np)

        objects_per_layer = defaultdict(list)

        for obj, label in zip(objects_in_shelf, labels):
            relative_obj = EGObject2D(
                id=obj.id,
                room_id=obj.room_id,
                place_id=obj.place_id,
                object_type=obj.object_type,
                scale=EGSize(
                    width=obj.scale.width,
                    length=obj.scale.length,
                    height=obj.scale.height,
                ),
                position=EGPoint2D(
                    x=obj.position.x - shelf.position.x,
                    y=obj.position.y - shelf.position.y,
                ),
                orientation=EGOrientation(
                    x=obj.orientation.x, y=obj.orientation.y, z=obj.orientation.z
                ),
                source_id=obj.source_id,
            )
            objects_per_layer[label].append(relative_obj)

        for _, objects in objects_per_layer.items():
            layer = EGShelfLayer(
                scale=EGSize(
                    width=shelf.scale.width, length=shelf.scale.length, height=0.02
                ),
                objects=objects,
            )
            shelf_layers.append(layer)

    return shelf_layers


def test_create_partnet_shelf(rclpy_node):
    loader = PartNetMobilityDatasetLoader()
    world = loader.load(41003)
    print(world.root)
    original_connections = world.connections
    doors = [body for body in world.bodies if not body.name.name.endswith("link_4")]
    with world.modify_world():
        shelf_corpus = Body(name=PrefixedName("map"))
        world.add_body(shelf_corpus)
        for body in world.bodies:
            if body.name.name.endswith("link_4"):
                for shape in body.collision.shapes:
                    body_in_shelf = Body(
                        name=PrefixedName(f"{shelf_corpus.name.name}_{str(shape)}")
                    )
                    body_in_shelf.collision.shapes = [shape]
                    c_shelf_body_in_shelf = FixedConnection(
                        parent=shelf_corpus, child=body_in_shelf
                    )
                    world.add_connection(c_shelf_body_in_shelf)

        for connection in original_connections:
            if isinstance(connection, RevoluteConnection):
                connection.parent = shelf_corpus
                print(connection.parent, connection.child)
                # for door in doors:
                #     c_door_new_shelf =
                #     door
                #
            else:
                world.remove_connection(connection)
                world.remove_kinematic_structure_entity(connection.parent)
                world.remove_kinematic_structure_entity(connection.child)

    revolute = [con for con in world.connections if isinstance(con, RevoluteConnection)]
    for con in revolute:
        con.position = 2

    print([body.name for body in world.bodies])
    assert len(world.bodies) > 0
    assert len(world.semantic_annotations) > 0

    marker = VizMarkerPublisher(node=rclpy_node, _world=world)
    marker.with_tf_publisher()


def test_rspn_fitting_on_shelves(rclpy_node):
    world = World()
    root = Body(name=PrefixedName(name="map"))

    with world.modify_world():
        world.add_body(root)

    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    if not session.scalars(select(EGObjectDAO)).first():
        session = add_to_database(session)

    shelf_layers = _extract_shelf_layers_from_objects(session)
    assert (
        shelf_layers
    ), "No shelves with objects found — check the sage-10k-layouts path."

    layer_daos = [to_dao(layer) for layer in shelf_layers]

    rspn = RelationalProbabilisticCircuit(EGShelfLayer)
    rspn = rspn.fit(layer_daos)

    assert rspn.class_probabilistic_circuit is not None
    assert rspn.class_probabilistic_circuit.is_valid()

    num_objects_per_layer = 3

    def _layer_query(n: int):
        return underspecified(EGShelfLayer)(
            scale=underspecified(EGSize)(width=..., length=..., height=...),
            objects=[
                underspecified(EGObject2D)(
                    id=None,
                    room_id=None,
                    place_id=None,
                    object_type=...,
                    scale=underspecified(EGSize)(width=..., length=..., height=...),
                    position=underspecified(EGPoint2D)(x=..., y=...),
                    orientation=underspecified(EGOrientation)(x=..., y=..., z=...),
                    source_id=None,
                )
                for _ in range(n)
            ],
        )

    registry = RelationalCircuitRegistry(
        relational_probabilistic_circuit=rspn, query=_layer_query(num_objects_per_layer)
    )
    prob_backend = ProbabilisticBackend(model_registry=registry, number_of_samples=1)

    sim = MujocoSim(world=world, step_size=0.000001, multiccd=True)
    sim.start_simulation()
    while sim.is_running():
        for i in range(10):
            sampled_layers = resolve_shelf_collisions(
                [
                    next(
                        iter(prob_backend.evaluate(_layer_query(num_objects_per_layer)))
                    )
                    for _ in range(4)
                ],
                rspn,
            )

            source_id_to_path = build_source_id_to_path()
            training_objects = session.scalars(
                select(EGObjectDAO).distinct().limit(1000)
            ).all()
            book_source_ids = [
                (source_id_to_path[obj.source_id], obj.source_id)
                for obj in training_objects
                if BookObjectType.contains(obj.object_type)
                and obj.source_id in source_id_to_path
            ]

            shelf_sample = EGShelf(
                position=EGPoint2D(x=0.0, y=0.0),
                scale=EGSize(height=2.0, length=1.5, width=0.5),
                orientation=EGOrientation(x=0.0, y=0.0, z=0.0),
                layers=sampled_layers,
                book_source_ids=book_source_ids,
            )

            assert all(layer.objects for layer in shelf_sample.layers)
            world = shelf_sample.create_in_world()
            sim.reload_world(world)
        sim.stop_simulation()
        viz_marker = VizMarkerPublisher(_world=world, node=rclpy_node)
        viz_marker.with_tf_publisher()
        time.sleep(10)


def test_book_z_matches_shelf_layer_z_regardless_of_book_height():
    """All books on the same layer must receive the same z, independent of book height.

    Book meshes have their origin at the bottom of the mesh, so the placement z is
    the layer board's top-surface z (z_height + layer_thickness/2).  Adding
    obj.scale.height/2 would shift taller books higher, making z inconsistent across books
    on the same layer.
    """
    layer_thickness = 0.02
    corpus_height = 2.0
    num_layers = 1

    layer_z_height = corpus_height / (num_layers + 1)
    expected_book_z = layer_z_height + layer_thickness / 2

    def make_book(book_id: str, height: float) -> EGObject2D:
        return EGObject2D(
            id=book_id,
            room_id="room1",
            place_id="shelf1",
            object_type=ObjectType.BOOK,
            scale=EGSize(width=0.1, length=0.05, height=height),
            position=EGPoint2D(x=0.0, y=0.0),
            orientation=EGOrientation(x=0.0, y=0.0, z=0.0),
            source_id="dummy_source",
        )

    shelf = EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGSize(height=corpus_height, length=0.4, width=0.4),
        orientation=EGOrientation(x=0.0, y=0.0, z=0.0),
        layers=[
            EGShelfLayer(
                scale=EGSize(width=0.4, length=0.4, height=layer_thickness),
                objects=[
                    make_book("short_book", height=0.15),
                    make_book("tall_book", height=0.40),
                ],
            )
        ],
        book_source_ids=[(Path("/dummy"), "dummy_source")],
    )

    with patch.object(EGObject2D, "create_in_world") as mock_create:
        shelf.create_in_world()

    assert mock_create.call_count == 2
    z_values = [call.kwargs["z"] for call in mock_create.call_args_list]
    assert z_values[0] == pytest.approx(expected_book_z), "short book z is wrong"
    assert z_values[1] == pytest.approx(expected_book_z), "tall book z is wrong"
