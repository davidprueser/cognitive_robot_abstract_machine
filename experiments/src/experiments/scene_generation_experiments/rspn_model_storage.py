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
    layers were coarsened against.

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
    The object types left unchanged when the training layers were coarsened; every other
    type was replaced with ``ObjectType.OTHER``.
    """

    @classmethod
    def load(cls, path: Path) -> TrainedArbitraryShelfModel:
        """
        Load a model previously exported with :meth:`save`.

        JSON has no set type, so the generic decoder restores
        ``frequent_object_types`` as a list; it is converted back to a set
        here to match the field's declared type.

        A model fitted before shelf types existed is rejected rather than used:
        it loads and samples perfectly well, so the only visible symptom would be
        every kind of shelf coming out the same.

        :param path: File to read the exported model from.
        :return: The restored model.
        :raises OutdatedTrainedModelError: If the fitted circuit predates the
            current schema.
        """
        restored = from_json(json.loads(path.read_text()))
        restored.frequent_object_types = set(restored.frequent_object_types)
        circuit = restored.relational_probabilistic_circuit
        modelled = {
            variable.name for variable in circuit.class_probabilistic_circuit.variables
        }
        if not any(name.endswith("shelf_type") for name in modelled):
            raise OutdatedTrainedModelError(
                model_path=str(path), missing_variable="shelf_type"
            )
        return restored

    def save(self, path: Path) -> None:
        """
        Export this model to *path* as JSON, creating parent directories as needed.

        :param path: File to write the exported model to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json(self)))
