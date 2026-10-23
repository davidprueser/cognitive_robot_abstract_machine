from experiments.lab_elevator_room_demo import (
    MAP_FRAME,
    RVizWorldPublisher,
    create_lab_elevator_room_world,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Elevator,
    Wall,
)


def test_world_is_rooted_in_the_map_frame():
    world = create_lab_elevator_room_world()
    assert str(world.root.name) == MAP_FRAME


def test_publisher_sends_a_marker_for_every_wall_and_the_elevator(rclpy_node):
    world = create_lab_elevator_room_world()
    publisher = RVizWorldPublisher(world=world, node=rclpy_node)
    marker_namespaces = {
        marker.ns for marker in publisher.marker_publisher.markers.markers
    }
    shown_annotations = world.get_semantic_annotations_by_type(
        Wall
    ) + world.get_semantic_annotations_by_type(Elevator)
    assert {
        str(annotation.root.name) for annotation in shown_annotations
    } <= marker_namespaces


def test_elevator_doors_stand_open():
    world = create_lab_elevator_room_world()
    (elevator,) = world.get_semantic_annotations_by_type(Elevator)
    assert [door.movable_joint.position for door in elevator.doors] == [
        door.movable_joint.dof.limits.upper.position for door in elevator.doors
    ]


def test_every_door_stands_open():
    world = create_lab_elevator_room_world()
    doors = world.get_semantic_annotations_by_type(Door)
    assert [door.movable_joint.position for door in doors] == [
        door.movable_joint.dof.limits.upper.position for door in doors
    ]


def test_publisher_resends_tf_and_markers_periodically(rclpy_node):
    publisher = RVizWorldPublisher(
        world=create_lab_elevator_room_world(), node=rclpy_node
    )
    assert sorted(timer.timer_period_ns for timer in rclpy_node.timers) == sorted(
        int(period * 1e9)
        for period in (publisher.tf_period_seconds, publisher.marker_period_seconds)
    )
