import os
from pathlib import Path
from unittest.mock import patch

import pytest

from sqlalchemy.orm import Session

from conftest import rclpy_node
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import (
    underspecified,
)
from krrood.ormatic.utils import create_engine
from krrood.parametrization.model_registries import (
    RelationalCircuitRegistry,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from semantic_digital_twin.scene_generation.scene_schema import (
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
)
from semantic_digital_twin.adapters.partnet_mobility_dataset.loader import (
    PartNetMobilityDatasetLoader,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.orm.ormatic_interface import *  # type: ignore
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body


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


def test_book_z_matches_shelf_layer_z_regardless_of_book_height():
    """
    All books on the same layer must receive the same z, independent of book
    height.

    Book meshes have their origin at the bottom of the mesh, so the
    placement z is the layer board's top-surface z (z_height +
    layer_thickness/2).  Adding obj.scale.height/2 would shift taller
    books higher, making z inconsistent across books on the same layer.
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
