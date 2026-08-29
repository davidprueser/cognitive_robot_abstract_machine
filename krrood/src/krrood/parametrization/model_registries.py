from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import Type, Dict

from krrood.parametrization.parameterizer import UnderspecifiedParameters
from krrood.utils import get_class_and_attribute_name
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_model import ProbabilisticModel


@dataclass
class ModelRegistry(ABC):
    """
    A registry that selects probabilistic models for given underspecified parameters of
    match-queries.
    """

    @abstractmethod
    def get_model(self, parameters: UnderspecifiedParameters) -> ProbabilisticModel:
        """
        :param parameters: The parameters to get a model for.
        :return: A probabilistic model that can be used to generate answers for the given expression.
        """


@dataclass
class FullyFactorizedRegistry(ModelRegistry):
    """
    A registry that always returns a fully factorized model.
    """

    def get_model(self, parameters: UnderspecifiedParameters) -> ProbabilisticModel:
        return fully_factorized(parameters.variables.values())


@dataclass
class DictRegistry(ModelRegistry):
    """
    A registry that uses a dictionary to keep all models.
    """

    models: Dict[Type, ProbabilisticModel]
    """
    A dictionary that maps classes to probabilistic models.
    """

    def get_model(self, parameters: UnderspecifiedParameters) -> ProbabilisticModel:
        return self.models[parameters.statement._expression.selected_variable._type_]


_ALTERNATIVE_MAPPING_ATTRIBUTE_ALIASES: tuple[tuple[str, str], ...] = (
    (".bearing", ".yaw"),
    (".position.x", ".x"),
    (".position.y", ".y"),
)
"""
Suffix rewrites from a circuit variable's fit-time name back to the domain attribute
name a query built directly against the domain class uses for the same field.

Fitting always goes through a type's DAO representation (JPT needs plain numeric
leaves, which a symbolic, ``casadi``-backed type like
:class:`~semantic_digital_twin.spatial_types.spatial_types.Pose2D` is not), and a
composite field whose DAO representation is an
:class:`~semantic_digital_twin.orm.model.AlternativeMapping` -- ``Pose2D``'s ``yaw`` is
stored as ``Pose2DMapping.bearing``, ``x``/``y`` nested a level deeper under
``position`` -- is therefore fit under names a query naming that field through the
domain class's own properties never produces. Without this rewrite such a field's
circuit variable is left under its fit-time name after grounding: a query
condition/truncation referencing it by its own domain name silently targets an
unconnected variable that never constrains anything, and reconstructing an instance
from a sample never finds a matching slot to write the sampled value into (see
:meth:`~krrood.parametrization.parameterizer.UnderspecifiedParameters.construct_instance_from_model_sample`),
leaving the field unset.

.. note::
    Scoped to the ``Pose2D``/``Point2`` mappings this project's spatial queries
    actually hit, not a general :class:`AlternativeMapping` field-name resolver -- a
    type registering a new ``AlternativeMapping`` with a differently-named field needs
    an entry added here too.
"""


def _alternative_mapping_domain_name(qualified_name: str) -> str:
    """
    Rewrite a circuit variable's fit-time (DAO/``AlternativeMapping``) name to the
    domain attribute name a query built against the domain class would use for the
    same field, if :data:`_ALTERNATIVE_MAPPING_ATTRIBUTE_ALIASES` names a rewrite for
    it.

    :param qualified_name: The circuit variable's fit-time name.
    :return: The domain-side name, or *qualified_name* unchanged if no rewrite applies.
    """
    for fit_time_suffix, domain_suffix in _ALTERNATIVE_MAPPING_ATTRIBUTE_ALIASES:
        if qualified_name.endswith(fit_time_suffix):
            return qualified_name[: -len(fit_time_suffix)] + domain_suffix
    return qualified_name


@dataclass
class RelationalCircuitRegistry(ModelRegistry):
    """
    A registry that grounds a RelationalProbabilisticCircuit for the queried statement
    and aligns its variable names to the UnderspecifiedParameters convention before
    returning.
    """

    relational_probabilistic_circuit: RelationalProbabilisticCircuit
    """
    The trained relational probabilistic circuit to ground.
    """

    def get_model(self, parameters: UnderspecifiedParameters) -> ProbabilisticModel:
        grounded = self.relational_probabilistic_circuit.ground(parameters.statement)
        class_prefix = self.relational_probabilistic_circuit.class_.__name__
        rename_map = {}
        for circuit_var in grounded.variables:
            # A circuit variable mounted from a nested exchangeable part (e.g. an
            # EGShelfLayer's "objects") already carries its own full, class-qualified
            # path by the time grounding is done with it (see
            # _rename_variables_with_part_prefix in the rspn module) -- prefixing it
            # again here would double it (e.g. "EGShelfLayer.EGShelfLayer.objects[0]
            # .pose.bearing"), which can never match anything in parameters.variables.
            # Only a variable straight off the queried class's own circuit -- never
            # renamed with a part prefix -- still needs class_prefix added.
            if circuit_var.name.startswith(f"{class_prefix}."):
                qualified_name = circuit_var.name
            else:
                qualified_name = get_class_and_attribute_name(
                    class_prefix, circuit_var.name
                )
            if qualified_name not in parameters.variables:
                qualified_name = _alternative_mapping_domain_name(qualified_name)
            if qualified_name in parameters.variables:
                rename_map[circuit_var] = parameters.variables[qualified_name]
        grounded.update_variables(rename_map)
        return grounded


@dataclass
class CausalCircuitRegistry(ModelRegistry):
    """
    A registry that maps target classes directly to pre-built causal circuits, so a
    ``cause``/``causes_effect()`` query can be routed through that circuit's
    ``backdoor_adjustment`` method.

    See
    :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`.
    """

    circuits: Dict[Type, CausalCircuit]
    """
    A dictionary that maps classes to pre-built causal circuits.
    """

    def get_model(self, parameters: UnderspecifiedParameters) -> ProbabilisticModel:
        return self.circuits[parameters.statement._expression.selected_variable._type_]
