from semantic_digital_twin.scene_generation.sage10k_processing import EGDataProcessing
from semantic_digital_twin.scene_generation.scene_schema import (
    SceneGenerator,
    EGRoom,
    EGSize,
    EGPosition,
    EGWall,
    EGPoint2D,
    EGDoor,
)
from semantic_digital_twin.world import World


def create_environment(scene_to_shelf_object: dict) -> tuple[SceneGenerator, World]:
    """
    Instantiate a SceneGenerator and its world from a mapping of scene ids to
    shelf objects.

    Downloads each scene's mesh assets if not already cached locally.

    :param scene_to_shelf_object: Mapping of scene id to shelf
        EGObjectDAO as returned by :func:`query_for_shelves`.
    :return: Tuple of (SceneGenerator, World).
    """
    data_processing = EGDataProcessing()
    scene_directory_to_object = {
        data_processing.download_specific_scene(scene_id): shelf_object
        for scene_id, shelf_object in scene_to_shelf_object.items()
    }
    mesh_to_object = {
        directory: shelf_object.from_dao()
        for directory, shelf_object in scene_directory_to_object.items()
    }
    scene_generator = SceneGenerator(
        id="scene_1",
        mesh_to_object_mapping=mesh_to_object,
        room=EGRoom(
            id="room_1",
            room_type="living_room",
            scale=EGSize(0, 1, 2),
            position=EGPosition(0, 0, 0),
            objects=list(mesh_to_object.values()),
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
    return scene_generator, world


if __name__ == "__main__":
    create_environment({})