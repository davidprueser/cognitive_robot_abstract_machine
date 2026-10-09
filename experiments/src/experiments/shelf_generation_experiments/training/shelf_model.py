from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from experiments.shelf_generation_experiments.shelf_schema import (
    RelationalCircuitExperimentShelf,
)
from experiments.shelf_generation_experiments.training.object_type_coarsening import (
    ObjectTypeCoarsening,
)
from krrood.adapters.json_serializer import from_json, to_json
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.ormatic.utils import classproperty
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)


# %% schema relations
class ShelfPart(StrEnum):
    """
    The exchangeable parts of the shelf schema, by the field holding them.
    """

    LAYERS = "layers"
    """
    The layers of a shelf.
    """

    OBJECTS = "objects"
    """
    The objects standing on a layer.
    """


# %% fitting
@dataclass
class ShelfModelFitSettings:
    """
    How a shelf model is fitted.
    """

    min_samples_per_leaf: float = 0.05
    """
    Smallest share of a level's training rows a leaf of the circuit of the shelf, of a
    layer and of an object holds, which allows at most ``1 / min_samples_per_leaf``
    leaves.

    Grounding copies a part's circuit once per part of a sampled shelf, so this bounds
    the memory a grounding needs.
    """

    min_samples_per_quantile: int = 200
    """
    Fewest training rows a histogram piece of a continuous variable describes.

    Finer histograms multiply the size of every leaf, which grounding copies once per
    part as well.
    """

    object_type_keep_count: int = 20
    """
    How many of the most frequent object types, and of the most frequent themes, the
    model tells apart.
    """

    def learning_method(self) -> JointProbabilityTree:
        """
        :return: A fresh learning method for one level of the shelf schema.
        """
        return JointProbabilityTree(min_samples_per_leaf=self.min_samples_per_leaf)


@dataclass
class ShelfModel:
    """
    A relational circuit over shelves, fitted on the preprocessed shelves, together with
    the object type coarsening it was fitted against.

    The two travel together: the circuit's object type domain is fixed by the types the
    coarsening kept, so a mesh pool or a theme coarsened differently would relabel types
    the circuit never saw.
    """

    circuit: RelationalProbabilisticCircuit
    """
    The fitted circuit over :class:`RelationalCircuitExperimentShelf`.
    """

    coarsening: ObjectTypeCoarsening
    """
    The object type coarsening the training shelves went through.
    """

    @classmethod
    def fit(
        cls,
        shelves: Sequence[RelationalCircuitExperimentShelf],
        settings: ShelfModelFitSettings | None = None,
    ) -> ShelfModel:
        """
        :param shelves: The preprocessed shelves to fit on.
        :param settings: How the circuit is fitted; the defaults of
            :class:`ShelfModelFitSettings` when omitted.
        :return: The model fitted on *shelves* after coarsening their object types.
        """
        settings = settings or ShelfModelFitSettings()
        coarsening = ObjectTypeCoarsening.from_shelves(
            shelves, keep_count=settings.object_type_keep_count
        )
        circuit = RelationalProbabilisticCircuit(
            RelationalCircuitExperimentShelf,
            learning_method=settings.learning_method(),
            part_learning_methods={
                part.value: settings.learning_method() for part in ShelfPart
            },
            min_samples_per_quantile=settings.min_samples_per_quantile,
        ).fit(coarsening.coarsen_shelves(shelves))
        return cls(circuit=circuit, coarsening=coarsening)

    @classproperty
    def variable_name_aliases(cls) -> dict[str, str]:
        """
        The names queries give the pose coordinates of an object, per name the circuit
        fitted them under; the circuit reads a pose through its mapping, which nests the
        coordinates in a position.
        """
        return {"pose.position.x": "pose.x", "pose.position.y": "pose.y"}

    @property
    def layer_circuit(self) -> RelationalProbabilisticCircuit:
        """
        The circuit over a single layer that the shelf circuit grounds once per layer.
        """
        return self.circuit.exchangeable_distribution_templates[
            ShelfPart.LAYERS
        ].template_distribution

    def shelf_backend(self, number_of_samples: int = 1) -> ProbabilisticBackend:
        """
        :param number_of_samples: How many samples an evaluation draws.
        :return: A backend answering queries over shelves.
        """
        return self._backend(self.circuit, number_of_samples)

    def layer_backend(self, number_of_samples: int = 1) -> ProbabilisticBackend:
        """
        :param number_of_samples: How many samples an evaluation draws.
        :return: A backend answering queries over single layers.
        """
        return self._backend(self.layer_circuit, number_of_samples)

    def _backend(
        self, circuit: RelationalProbabilisticCircuit, number_of_samples: int
    ) -> ProbabilisticBackend:
        """
        :param circuit: The circuit to answer queries with.
        :param number_of_samples: How many samples an evaluation draws.
        :return: A backend over *circuit* aligning its pose names with queries.
        """
        return ProbabilisticBackend(
            model_registry=RelationalCircuitRegistry(
                relational_probabilistic_circuit=circuit,
                variable_name_aliases=self.variable_name_aliases,
            ),
            number_of_samples=number_of_samples,
        )

    @classmethod
    def load(cls, path: Path) -> ShelfModel:
        """
        :param path: File a model was saved to with :meth:`save`.
        :return: The stored model.
        """
        return from_json(json.loads(path.read_text()))

    def save(self, path: Path) -> None:
        """
        Store this model as JSON at *path*, creating missing parent directories.

        :param path: File to store the model in.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json(self)))
