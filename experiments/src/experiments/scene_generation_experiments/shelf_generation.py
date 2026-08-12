from __future__ import annotations

import dataclasses
import enum
import json
import os
import time
from collections import Counter
from pathlib import Path

from sqlalchemy.orm import Session

from krrood.adapters.json_serializer import from_json, to_json
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from krrood.utils import get_full_class_name
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    UnivariateDiscreteLeaf,
)
from probabilistic_model.utils import MissingDict

from experiments.orm.ormatic_interface import *  # type: ignore
from experiments.scene_generation_experiments.book_shelf_generation import (
    _extract_shelf_layers_from_place_id,
)
from experiments.scene_generation_experiments.utils import (
    _get_source_ids_for_objects,
    load_all_objects,
    rclpy_node,
    min_samples_per_leaf_for,
)
from experiments.scene_generation_experiments.collision_resolution import (
    build_free_layer_query,
    build_layer_query_with_fixed_scale,
)
from experiments.scene_generation_experiments.in_world_resolver import (
    InWorldLayoutResolver,
)
from experiments.scene_generation_experiments.rspn_sampling import probabilistic_backend
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.scene_generation.scene_schema import (
    EGPoint2D,
    EGRotation,
    EGShelf,
    EGShelfLayer,
    EGScale,
    MeshCandidate,
    ObjectType,
)


def _frequent_object_types(
    shelf_layers: list[EGShelfLayer],
    keep_count: int,
) -> set[ObjectType]:
    """
    Return the *keep_count* most frequent object types across all objects in
    *shelf_layers*.

    :param shelf_layers: Layers whose objects' types are counted.
    :param keep_count: Number of distinct, most frequent object types to
        return.
    :return: The most frequent object types.
    """
    type_counts = Counter(
        object_2d.object_type for layer in shelf_layers for object_2d in layer.objects
    )
    return {object_type for object_type, _ in type_counts.most_common(keep_count)}


def _coarsen_rare_object_types(
    shelf_layers: list[EGShelfLayer],
    keep_count: int = 20,
) -> list[EGShelfLayer]:
    """
    Return new shelf layers where every object's type outside the *keep_count*
    most frequent types (across all objects in *shelf_layers*) is replaced with
    ``ObjectType.OTHER``.

    :param shelf_layers: Layers whose objects' types should be
        coarsened.
    :param keep_count: Number of distinct, most frequent object types to
        leave unchanged.
    :return: New EGShelfLayer instances with coarsened object types; all
        other fields (position, scale, orientation, source_id, ...) are
        unchanged.
    """
    frequent_types = _frequent_object_types(shelf_layers, keep_count)
    return [
        dataclasses.replace(
            layer,
            objects=[
                (
                    object_2d
                    if object_2d.object_type in frequent_types
                    else dataclasses.replace(object_2d, object_type=ObjectType.OTHER)
                )
                for object_2d in layer.objects
            ],
        )
        for layer in shelf_layers
    ]


def _coarsen_mesh_candidate_types(
    candidates: list[MeshCandidate],
    frequent_types: set[ObjectType],
) -> list[MeshCandidate]:
    """
    Return new mesh candidates where every candidate whose type falls outside
    *frequent_types* is relabeled as ``ObjectType.OTHER``.

    Mirrors :func:`_coarsen_rare_object_types` so the mesh pool's type labels
    line up with the coarsened types the RSPN actually samples -- without
    this, a sampled ``ObjectType.OTHER`` object would never find a same-type
    mesh candidate, since every candidate still carries its original,
    uncoarsened type.

    :param candidates: Mesh candidates whose types should be coarsened.
    :param frequent_types: Object types to leave unchanged; every other
        type is replaced with ``ObjectType.OTHER``.
    :return: New candidates with coarsened types.
    """
    return [
        (
            candidate
            if candidate.object_type in frequent_types
            else dataclasses.replace(candidate, object_type=ObjectType.OTHER)
        )
        for candidate in candidates
    ]


DEFAULT_ARBITRARY_SHELF_MODEL_PATH = (
    Path.home() / "Documents" / "sage-10k-models" / "arbitrary_shelf_rspn.json"
)
"""
Where :func:`generate_shelf_with_arbitrary_objects` reads and writes its
exported :class:`TrainedArbitraryShelfModel`.
"""


def _categorical_leaves(
    circuit: RelationalProbabilisticCircuit,
) -> list[UnivariateDiscreteLeaf]:
    """
    Collect every enum-valued leaf in *circuit*'s class circuit and,
    recursively, every nested exchangeable part's circuit.

    A leaf's own :attr:`~UnivariateLeaf.variable` distinguishes an
    enum-valued (``Symbolic``, backed by a :class:`~random_events.set.Set`)
    leaf from an integer-valued one (backed by a
    :class:`~random_events.interval.Interval`, e.g. an aggregation count),
    which this excludes since it needs no cross-process fix-up.

    :param circuit: The relational circuit to search.
    :return: Every enum-valued leaf found.
    """
    leaves = [
        node
        for node in (
            circuit.class_probabilistic_circuit.nodes()
            if circuit.class_probabilistic_circuit is not None
            else []
        )
        if isinstance(node, UnivariateDiscreteLeaf)
        and hasattr(node.variable.domain, "all_elements")
    ]
    for template in circuit.exchangeable_distribution_templates.values():
        leaves.extend(_categorical_leaves(template.template_distribution))
    return leaves


def _categorical_hash_registry(
    circuit: RelationalProbabilisticCircuit,
) -> dict[str, int]:
    """
    Record ``hash(member)``, as computed in the current process, for every
    enum member appearing in any of *circuit*'s enum-valued leaves.

    :param circuit: The relational circuit to inspect.
    :return: Mapping from ``"<enum class>#<member name>"`` to that member's
        hash in the current process.
    """
    return {
        f"{get_full_class_name(type(member))}#{member.name}": hash(member)
        for leaf in _categorical_leaves(circuit)
        for member in leaf.variable.domain.all_elements
        if isinstance(member, enum.Enum)
    }


def _restore_categorical_hashes(
    circuit: RelationalProbabilisticCircuit, saved_registry: dict[str, int]
) -> None:
    """
    Rewrite every enum-valued leaf's probability-table keys from the hashes
    *saved_registry* recorded to this process's own hashes for the same enum
    members.

    Python randomizes ``hash()`` for ``str``-backed types -- which includes
    a :class:`~enum.StrEnum` such as :class:`ObjectType` -- independently per
    process, so a leaf's fitted probabilities, keyed by ``hash(member)`` from
    whichever process fit *circuit*, would otherwise no longer match any
    domain element once evaluated in a different process, such as after
    loading an exported model.

    :param circuit: The relational circuit to fix up, mutated in place.
    :param saved_registry: The registry :func:`_categorical_hash_registry`
        produced in the process that fit *circuit*.
    """
    for leaf in _categorical_leaves(circuit):
        translation = {
            saved_registry[key]: hash(member)
            for member in leaf.variable.domain.all_elements
            if isinstance(member, enum.Enum)
            and (key := f"{get_full_class_name(type(member))}#{member.name}")
            in saved_registry
        }
        if not translation:
            continue
        leaf.distribution.probabilities = MissingDict(
            float,
            {
                translation.get(key, key): probability
                for key, probability in leaf.distribution.probabilities.items()
            },
        )


@dataclasses.dataclass
class TrainedArbitraryShelfModel:
    """
    A fitted arbitrary-shelf RSPN paired with the frequent object types its
    training layers were coarsened against.

    The two must always travel together: the circuit's ``ObjectType`` domain
    is fixed by which types :func:`_coarsen_rare_object_types` kept at fit
    time, so a mesh pool coarsened against a different ``frequent_object_types``
    set would relabel types the circuit never saw, raising a domain mismatch
    when the model is used later.
    """

    relational_probabilistic_circuit: RelationalProbabilisticCircuit
    """
    The fitted RSPN over :class:`EGShelfLayer`.
    """

    frequent_object_types: set[ObjectType]
    """
    The object types left unchanged when the training layers were coarsened;
    every other type was replaced with ``ObjectType.OTHER``.
    """

    categorical_hash_registry: dict[str, int] = dataclasses.field(default_factory=dict)
    """
    Populated by :meth:`save` from the fitting process's own ``hash()`` of
    every enum member the circuit's leaves reference, and used by
    :meth:`load` to keep those leaves' probability tables addressable after
    a cross-process reload. See :func:`_restore_categorical_hashes`.
    """

    @classmethod
    def load(cls, path: Path) -> TrainedArbitraryShelfModel:
        """
        Load a model previously exported with :meth:`save`.

        JSON has no set type, so the generic decoder restores
        ``frequent_object_types`` as a list; it is converted back to a set
        here to match the field's declared type.

        :param path: File to read the exported model from.
        :return: The restored model.
        """
        restored = from_json(json.loads(path.read_text()))
        restored.frequent_object_types = set(restored.frequent_object_types)
        _restore_categorical_hashes(
            restored.relational_probabilistic_circuit,
            restored.categorical_hash_registry,
        )
        return restored

    def save(self, path: Path) -> None:
        """
        Export this model to *path* as JSON, creating parent directories as
        needed.

        :param path: File to write the exported model to.
        """
        self.categorical_hash_registry = _categorical_hash_registry(
            self.relational_probabilistic_circuit
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json(self)))


def generate_shelf_with_arbitrary_objects(
    node, model_path: Path = DEFAULT_ARBITRARY_SHELF_MODEL_PATH
) -> None:
    """
    Train an RSPN on all object types found on shelves in the dataset and
    visualise a sampled, collision-free arrangement via RViz.

    Unlike :func:`book_shelf_generation.generate_book_shelf`, this demo
    includes every object type found on shelves in the training data — books,
    cups, plants, containers, and more — so the RSPN learns the joint
    spatial distribution across all of them. Mesh assets are drawn at random
    from the pool of available shelf-object PLY files that share the same
    (generalized) object type as the object sampled by the RSPN; if no mesh
    of that type is available, a mesh is drawn from the full pool instead.

    .. note::
        Meshes are rescaled so their bounding box matches the RSPN-sampled
        scale, and collisions are resolved against those real meshes. A mesh
        whose native aspect ratio differs from the sampled scale is stretched
        to fit, which can look unnatural for high-variance types (e.g. plants,
        containers).

    :param node: An active rclpy node used to publish visualisation markers.
    :param model_path: Where the fitted model is exported to and, on a later
        run, loaded from instead of being refit. Training data is only
        queried and the RSPN only fit when no model exists at this path yet.
    """
    start = time.time()
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    engine = create_engine(uri)
    Base.metadata.create_all(bind=engine)
    session = Session(engine)

    if model_path.exists():
        trained_model = TrainedArbitraryShelfModel.load(model_path)
    else:
        shelf_layers, _ = _extract_shelf_layers_from_place_id(session, object_type=None)
        frequent_types = _frequent_object_types(shelf_layers, keep_count=20)
        shelf_layers = _coarsen_rare_object_types(shelf_layers)
        shelf_layer_data_access_objects = [to_dao(layer) for layer in shelf_layers]

        rspn = RelationalProbabilisticCircuit(
            EGShelfLayer,
            min_samples_per_leaf=min_samples_per_leaf_for(
                sum(len(layer.objects) for layer in shelf_layers)
            ),
        ).fit(shelf_layer_data_access_objects)

        trained_model = TrainedArbitraryShelfModel(
            relational_probabilistic_circuit=rspn,
            frequent_object_types=frequent_types,
        )
        trained_model.save(model_path)

    rspn = trained_model.relational_probabilistic_circuit
    frequent_types = trained_model.frequent_object_types

    probability_backend = probabilistic_backend(rspn)

    objects_per_layer = 3
    layer_count = 4
    reference_layer = next(
        iter(probability_backend.evaluate(build_free_layer_query(objects_per_layer)))
    )
    target_scale = reference_layer.scale
    remaining_layers = [
        next(
            iter(
                probability_backend.evaluate(
                    build_layer_query_with_fixed_scale(objects_per_layer, target_scale)
                )
            )
        )
        for _ in range(layer_count - 1)
    ]
    sampled_layers = [reference_layer] + remaining_layers

    source_ids = _get_source_ids_for_objects(
        load_all_objects(session), object_type=None
    )
    source_ids = _coarsen_mesh_candidate_types(source_ids, frequent_types)
    shelf_sample = EGShelf(
        position=EGPoint2D(x=0.0, y=0.0),
        scale=EGScale(height=2.0, length=target_scale.length, width=target_scale.width),
        orientation=EGRotation(x=0.0, y=0.0, z=0.0),
        layers=sampled_layers,
        source_ids=source_ids,
    )

    spawned_shelf = InWorldLayoutResolver.for_shelf(shelf_sample, rspn).resolve()
    world = spawned_shelf.world
    viz_marker = VizMarkerPublisher(_world=world, node=node)
    viz_marker.with_tf_publisher()
    print(f"Finished generating shelf sample in {time.time() - start:.2f}s")


if __name__ == "__main__":
    with rclpy_node() as node:
        generate_shelf_with_arbitrary_objects(node)
