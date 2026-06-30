from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sqlalchemy import select
from sqlalchemy.orm import Session

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import (
    a,
    set_of,
    variable,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState
from krrood.ormatic.utils import create_engine
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)

from experiments.orm.ormatic_interface import *  # type: ignore
from experiments.scene_generation_experiments.collision_resolution import (
    build_free_layer_query,
    resolve_shelf_collisions,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.scene_generation.sage10k_processing import EGDataProcessing
from semantic_digital_twin.scene_generation.scene_schema import (
    BookObjectType,
    EGDoor,
    EGObject2D,
    EGOrientation,
    EGPoint2D,
    EGPosition,
    EGRoom,
    EGShelf,
    EGShelfLayer,
    EGSize,
    EGWall,
    ObjectType,
    SceneGenerator,
    build_source_id_to_path,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.utils import rclpy_installed
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.world_entity import Body

def load_scenes_from_layout_directory(path: Path) -> list[SceneGenerator]:
    """
    Parse all layout JSON files under *path* and return the corresponding
    scenes.

    Each layout sub-directory is expected to contain exactly one
    ``layout_*.json`` file.

    :param path: Root directory that contains layout sub-directories.
    :return: List of parsed :class:`SceneGenerator` instances.
    """
    json_directories = list(path.glob("*layout*"))
    json_files = []
    for json_directory in json_directories:
        [json_file] = list(json_directory.glob("layout_*.json"))
        json_files.append(json_file)
    scenes = []
    for file in json_files:
        json_dict = json.loads(file.read_text())
        scene = SceneGenerator._from_json(json_dict)
        scene.directory = path
        scenes.append(scene)
    return scenes


def add_to_database(session: Session) -> Session:
    """
    Load the first 100 sage-10k scenes from the local layouts directory into
    the database.

    :param session: An active SQLAlchemy session bound to an initialised
        schema.
    :return: The session after committing all scene records.
    :raises ValueError: If the layouts directory does not exist.
    """
    path = Path.home() / "Downloads" / "sage-10k-layouts"
    if not path.exists():
        raise ValueError(f"Path {path} does not exist")
    data_access_object_state = ToDataAccessObjectState()
    scene_data_access_objects = []
    for scene in load_scenes_from_layout_directory(path)[0:100]:
        scene_dao = to_dao(scene, data_access_object_state)
        scene_data_access_objects.append(scene_dao)
        session.add(scene_dao)
    session.commit()
    return session


def query_for_shelves(session: Session) -> dict:
    """
    Query the database for scenes paired with one shelf object each.

    :param session: An active SQLAlchemy session.
    :return: Mapping of scene id to the associated shelf EGObjectDAO.
    """
    scenes = session.scalars(select(SceneGeneratorDAO)).all()
    objects = session.scalars(select(EGObjectDAO)).all()

    scene_variable = variable(SceneGeneratorDAO, scenes)
    object_variable = variable(EGObjectDAO, objects)

    query = (
        a(set_of(scene_variable.id, object_variable))
        .where(object_variable.object_type == "shelf")
        .where(scene_variable.room.id == object_variable.room_id)
        .distinct()
        .limit(5)
        .tolist()
    )
    return {row[scene_variable.id]: row[object_variable] for row in query}


def query_environments(session: Session) -> tuple:
    """
    Return all scene and object records from the database.

    :param session: An active SQLAlchemy session.
    :return: Tuple of (scenes, objects) as SQLAlchemy scalars.
    """
    scenes = session.scalars(select(SceneGeneratorDAO)).all()
    objects = session.scalars(select(EGObjectDAO)).all()
    return scenes, objects


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


def _extract_shelf_layers_from_objects(
        session: Session, edge_margin_fraction: float = 0.10
) -> tuple[list[EGShelfLayer], list[EGObjectDAO]]:
    """
    Load all scenes and group objects by the shelf they are placed on.

    Uses bounding-box spatial reasoning (not place_id alone) so the
    grouping mirrors what a robot would perceive from geometry rather
    than metadata.

    :param edge_margin_fraction: Fraction of each shelf dimension to use
        as an inset margin on X and Y. Objects whose centre falls within
        this margin of the shelf boundary are excluded from training
        data so the learned RSPN does not place books at positions where
        they would fall off in simulation.
    """
    if not session.scalars(select(EGObjectDAO)).first():
        session = add_to_database(session)

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

        z_positions = [obj.position.z for obj in objects_in_shelf]
        z_positions_array = np.array(z_positions).reshape(-1, 1)

        labels = DBSCAN(
            eps=0.05,
            min_samples=1,
        ).fit_predict(z_positions_array)

        objects_per_layer: defaultdict[int, list] = defaultdict(list)

        for obj, label in zip(objects_in_shelf, labels):
            relative_object = EGObject2D(
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
            objects_per_layer[label].append(relative_object)

        for _, layer_objects in objects_per_layer.items():
            layer = EGShelfLayer(
                scale=EGSize(
                    width=shelf.scale.width, length=shelf.scale.length, height=0.02
                ),
                objects=layer_objects,
            )
            shelf_layers.append(layer)

    return shelf_layers, objects


def _extract_shelf_layers_from_place_id(
    session: Session,
    edge_margin_fraction: float = 0.10,
    type_predicate: Callable[[ObjectType], bool] = BookObjectType.contains,
) -> tuple[list[EGShelfLayer], list[EGObjectDAO]]:
    """
    Load all scenes and group objects by the shelf declared in their
    ``place_id``.

    An object is considered a shelf occupant when ``"shelf"`` appears in its
    ``place_id`` (e.g. ``room_b12d7278_shelf_51fd4e1e``).  Shelf membership
    is determined purely from the dataset metadata rather than spatial
    bounding-box containment.

    After grouping, objects whose centre falls outside the shelf's XY footprint
    (inset by *edge_margin_fraction*) are discarded so that the learned RSPN
    does not place objects at positions where they would protrude from the shelf.

    :param edge_margin_fraction: Fraction of each shelf dimension to use as
        an inset margin on X and Y when filtering out-of-bounds objects.
    :param type_predicate: Called with each object's :class:`ObjectType`; only
        objects for which this returns ``True`` are included. Defaults to
        :meth:`BookObjectType.contains` to reproduce the original book-only
        behaviour.
    :return: Extracted shelf layers and all loaded object DAOs.
    """
    if not session.scalars(select(EGObjectDAO)).first():
        session = add_to_database(session)

    objects = session.scalars(select(EGObjectDAO).distinct().limit(50000)).all()

    shelf_by_id: dict[str, EGObjectDAO] = {
        obj.id: obj for obj in objects if obj.object_type == ObjectType.SHELF
    }

    shelf_objects = [
        obj
        for obj in objects
        if type_predicate(obj.object_type) and "shelf" in obj.place_id
    ]

    objects_per_shelf: defaultdict[str, list[EGObjectDAO]] = defaultdict(list)
    for obj in shelf_objects:
        objects_per_shelf[obj.place_id].append(obj)

    shelf_layers = []
    for shelf_id, members in objects_per_shelf.items():
        shelf = shelf_by_id.get(shelf_id)
        if shelf is None:
            continue

        max_relative_x = shelf.scale.width / 2 * (1 - edge_margin_fraction)
        max_relative_y = shelf.scale.length / 2 * (1 - edge_margin_fraction)

        within_bounds = [
            obj for obj in members
            if abs(obj.position.x - shelf.position.x) <= max_relative_x
            and abs(obj.position.y - shelf.position.y) <= max_relative_y
        ]
        if not within_bounds:
            continue

        z_positions = np.array([obj.position.z for obj in within_bounds]).reshape(-1, 1)
        labels = DBSCAN(eps=0.05, min_samples=1).fit_predict(z_positions)

        objects_per_layer: defaultdict[int, list[EGObject2D]] = defaultdict(list)
        for obj, label in zip(within_bounds, labels):
            relative_object = EGObject2D(
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
            objects_per_layer[label].append(relative_object)

        for _, layer_objects in objects_per_layer.items():
            shelf_layers.append(
                EGShelfLayer(
                    scale=EGSize(
                        width=shelf.scale.width, length=shelf.scale.length, height=0.02
                    ),
                    objects=layer_objects,
                )
            )

    return shelf_layers, objects


@contextlib.contextmanager
def rclpy_node():
    """
    Context manager that initialises an rclpy node and spins it in a background
    thread.

    :raises ValueError: If rclpy is not installed.
    """
    if not rclpy_installed():
        raise ValueError("No ros installed")
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("test_node")

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
    thread.start()
    time.sleep(0.1)
    try:
        yield node
    finally:
        executor.shutdown()
        thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


def _get_source_ids_for_objects(
    objects: list[EGObjectDAO],
    type_predicate: Callable[[ObjectType], bool] = BookObjectType.contains,
) -> list[tuple[Path, str]]:
    """
    Extract all (scene_dir, source_id) pairs for objects accepted by
    *type_predicate* that have a local PLY mesh available.

    :param objects: All loaded object DAOs from the database.
    :param type_predicate: Called with each object's
        :class:`ObjectType`; only objects for which this returns
        ``True`` are included. Defaults to
        :meth:`BookObjectType.contains` to reproduce the original book-
        only behaviour.
    :return: List of (scene_directory, source_id) pairs.
    """
    source_id_to_path = build_source_id_to_path()
    return [
        (source_id_to_path[obj.source_id], obj.source_id)
        for obj in objects
        if type_predicate(obj.object_type) and obj.source_id in source_id_to_path
    ]

def generate_book_shelf(node) -> None:
    """
    Train an RSPN on shelf-layer data from the database, sample collision-free
    book arrangements, and visualise them via RViz markers.

    :param node: An active rclpy node used to publish visualisation
        markers.
    """
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    # shelf_layers, training_objects = _extract_shelf_layers_from_objects(session)
    shelf_layers, training_objects = _extract_shelf_layers_from_place_id(session)
    shelf_layer_data_access_objects = [to_dao(layer) for layer in shelf_layers]

    rspn = RelationalProbabilisticCircuit(EGShelfLayer)
    rspn = rspn.fit(shelf_layer_data_access_objects)

    registry = RelationalCircuitRegistry(
        relational_probabilistic_circuit=rspn
    )
    probability_backend = ProbabilisticBackend(model_registry=registry, number_of_samples=1)

    sampled_layers = resolve_shelf_collisions(
        [
            next(iter(probability_backend.evaluate(build_free_layer_query(3))))
            for _ in range(4)
        ],
        rspn,
    )

    source_ids_for_sampled_objects = _get_source_ids_for_objects(training_objects)
    shelf_sample = EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGSize(height=2.0, length=1.5, width=0.5),
        orientation=EGOrientation(x=0.0, y=0.0, z=0.0),
        layers=sampled_layers,
        book_source_ids=source_ids_for_sampled_objects,
    )

    world = shelf_sample.create_in_world()
    viz_marker = VizMarkerPublisher(_world=world, node=node)
    viz_marker.with_tf_publisher()


if __name__ == "__main__":
    with rclpy_node() as node:
        generate_book_shelf(node)