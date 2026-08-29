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
