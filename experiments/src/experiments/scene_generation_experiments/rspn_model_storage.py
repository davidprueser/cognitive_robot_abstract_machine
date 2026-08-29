from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from krrood.adapters.json_serializer import from_json, to_json
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from experiments.scene_generation_experiments.exceptions import (
    OutdatedTrainedModelError,
)
from semantic_digital_twin.scene_generation.scene_schema import ObjectType


@dataclasses.dataclass
class TrainedArbitraryShelfModel:
    """
    A fitted arbitrary-shelf RSPN paired with the frequent object types its training
    layers, and the frequent themes its training shelves, were coarsened against.

    All three must always travel together: the circuit's ``ObjectType`` domain is
    fixed by which types :func:`_coarsen_rare_object_types` and
    :func:`_coarsen_rare_shelf_themes` kept at fit time, so a mesh pool or a
    requested theme coarsened against a different frequent-types set would relabel
    types the circuit never saw, raising a domain mismatch when the model is used
    later.
    """

    relational_probabilistic_circuit: RelationalProbabilisticCircuit
    """
    The fitted RSPN over :class:`EGShelfLayer`.
    """

    frequent_object_types: set[ObjectType]
    """
    The object types left unchanged when the training layers were coarsened; every other
    type was replaced with ``ObjectType.OTHER``.
    """

    frequent_theme_types: set[ObjectType] = dataclasses.field(default_factory=set)
    """
    The dominant types left unchanged when the training shelves' themes were
    coarsened; every other theme was replaced with ``ObjectType.OTHER``.

    A separate set from :attr:`frequent_object_types`: a type common on individual
    objects is not necessarily common as a shelf's own mode, so the two frequency
    counts are taken independently.
    """

    @classmethod
    def load(cls, path: Path) -> TrainedArbitraryShelfModel:
        """
        Load a model previously exported with :meth:`save`.

        JSON has no set type, so the generic decoder restores
        ``frequent_object_types``/``frequent_theme_types`` as lists; they are
        converted back to sets here to match the fields' declared types.

        A model fitted before themes existed is rejected rather than used: it
        loads and samples perfectly well, so the only visible symptom would be
        every theme coming out the same.

        :param path: File to read the exported model from.
        :return: The restored model.
        :raises OutdatedTrainedModelError: If the fitted circuit predates the
            current schema.
        """
        restored = from_json(json.loads(path.read_text()))
        restored.frequent_object_types = set(restored.frequent_object_types)
        restored.frequent_theme_types = set(restored.frequent_theme_types)
        circuit = restored.relational_probabilistic_circuit
        modelled = {
            variable.name for variable in circuit.class_probabilistic_circuit.variables
        }
        if not any(name.endswith("theme_dominant_type") for name in modelled):
            raise OutdatedTrainedModelError(
                model_path=str(path), missing_variable="theme_dominant_type"
            )
        return restored

    def save(self, path: Path) -> None:
        """
        Export this model to *path* as JSON, creating parent directories as needed.

        :param path: File to write the exported model to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json(self)))
