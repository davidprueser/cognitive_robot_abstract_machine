from __future__ import annotations

import dataclasses
import enum
import json
from pathlib import Path

from krrood.adapters.json_serializer import from_json, to_json
from krrood.utils import get_full_class_name
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    UnivariateDiscreteLeaf,
)
from probabilistic_model.utils import MissingDict
from experiments.scene_generation_experiments.exceptions import (
    OutdatedTrainedModelError,
)
from semantic_digital_twin.scene_generation.scene_schema import ObjectType


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
        _restore_categorical_hashes(
            restored.relational_probabilistic_circuit,
            restored.categorical_hash_registry,
        )
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
        Export this model to *path* as JSON, creating parent directories as
        needed.

        :param path: File to write the exported model to.
        """
        self.categorical_hash_registry = _categorical_hash_registry(
            self.relational_probabilistic_circuit
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json(self)))


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
