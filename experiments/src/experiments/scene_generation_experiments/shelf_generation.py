from __future__ import annotations

import os

from sqlalchemy.orm import Session

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)

from experiments.orm.ormatic_interface import *  # type: ignore
from experiments.scene_generation_experiments.book_shelf_generation import (
    _extract_shelf_layers_from_place_id,
)
from experiments.scene_generation_experiments.utils import rclpy_node, _get_source_ids_for_objects
from experiments.scene_generation_experiments.collision_resolution import (
    build_free_layer_query,
    build_layer_query_with_fixed_scale,
    resolve_shelf_collisions,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.scene_generation.scene_schema import (
    EGOrientation,
    EGPoint2D,
    EGShelf,
    EGShelfLayer,
    EGSize,
    ObjectType,
)


def _is_shelf_furniture(object_type: ObjectType) -> bool:
    """
    Return ``True`` when *object_type* represents a shelf unit (furniture),
    ``False`` for any object that can be placed on a shelf.
    """
    return object_type == ObjectType.SHELF


def generate_shelf_with_arbitrary_objects(node) -> None:
    """
    Train an RSPN on all non-shelf object types found on shelves in the dataset
    and visualise a sampled, collision-free arrangement via RViz.

    Unlike :func:`book_shelf_generation.generate_book_shelf`, this demo
    includes every object type found on shelves in the training data — books,
    cups, plants, containers, and more — so the RSPN learns the joint
    spatial distribution across all of them. Mesh assets are drawn randomly
    from the full pool of available shelf-object PLY files, which means the
    rendered mesh type may not match the object type sampled by the RSPN;
    this is intentional for this demo.

    .. note::
        The RSPN learns object *scale* from training data, but PLY meshes are
        rendered at their native size. Object types with high size variance
        (e.g. plants, containers) may produce visual overlaps even after
        collision resolution.

    :param node: An active rclpy node used to publish visualisation markers.
    """
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    predicate = lambda object_type: not _is_shelf_furniture(object_type)

    shelf_layers, training_objects = _extract_shelf_layers_from_place_id(
        session, type_predicate=predicate
    )
    shelf_layer_data_access_objects = [to_dao(layer) for layer in shelf_layers]

    rspn = RelationalProbabilisticCircuit(EGShelfLayer)
    rspn = rspn.fit(shelf_layer_data_access_objects)

    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=rspn)
    probability_backend = ProbabilisticBackend(model_registry=registry, number_of_samples=1)

    reference_layer = next(iter(probability_backend.evaluate(build_free_layer_query(3))))
    target_scale = reference_layer.scale
    remaining_layers = [
        next(iter(probability_backend.evaluate(build_layer_query_with_fixed_scale(3, target_scale))))
        for _ in range(3)
    ]
    sampled_layers = resolve_shelf_collisions([reference_layer] + remaining_layers, rspn)

    source_ids = _get_source_ids_for_objects(training_objects, type_predicate=predicate)
    shelf_sample = EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGSize(height=2.0, length=target_scale.length, width=target_scale.width),
        orientation=EGOrientation(x=0.0, y=0.0, z=0.0),
        layers=sampled_layers,
        book_source_ids=source_ids,  # field name is legacy; pool now covers all types
    )

    world = shelf_sample.create_in_world()
    viz_marker = VizMarkerPublisher(_world=world, node=node)
    viz_marker.with_tf_publisher()


if __name__ == "__main__":
    with rclpy_node() as node:
        generate_shelf_with_arbitrary_objects(node)
