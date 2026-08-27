import enum
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from types import NoneType
from typing import Iterable, TypeVar

import numpy as np
from sqlalchemy.orm import sessionmaker
from typing_extensions import ClassVar, Dict, Optional

from krrood import logger
from krrood.entity_query_language.verbalization.vocabulary.english import Directive

from krrood.entity_query_language.core.base_expressions import (
    Selectable,
    SymbolicExpression,
)
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.exceptions import (
    NoSolutionFound,
    GenerativeBackendQueryIsNotUnderspecifiedVariable,
    SelectiveBackendCannotResolveEllipsisMatch,
    UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration,
)
from krrood.entity_query_language.factories import entity, set_of, variable
from krrood.entity_query_language.query.match import Match, AttributeMatch
from krrood.entity_query_language.query.query import Query
from krrood.ormatic.eql_interface import eql_to_sql
from probabilistic_model.exceptions import IntractableError
from probabilistic_model.probabilistic_circuit.rx.helper import uniform_measure_of_event
from probabilistic_model.probabilistic_model import ProbabilisticModel

try:
    from krrood.parametrization.model_registries import (
        ModelRegistry,
        FullyFactorizedRegistry,
    )
    from krrood.parametrization.parameterizer import (
        UnderspecifiedParameters,
    )
except ImportError as e:
    logger.debug(f"Couldn't import probabilistic model needed classes: {e}")
    ModelRegistry = NoneType
    FullyFactorizedRegistry = NoneType
    UnderspecifiedParameters = NoneType

T = TypeVar("T")


@dataclass
class QueryBackend(ABC):
    """
    Base class for all query backends.

    Query backends are objects that answer queries by different means.
    """

    opening_directive: ClassVar[Optional[Directive]] = None
    """
    The opening verb a verbalization uses when this backend evaluates the expression
    (``None`` keeps the query-type default).

    A backend declares its own performative so the verbalization layer never inspects
    concrete backend types.
    """

    @abstractmethod
    def evaluate(self, expression: Evaluable) -> Iterable[T]:
        """
        Generate answers that match the expression.

        :param expression: The expression to generate answers for.
        :return: An iterable of answers.
        """


@dataclass
class SelectiveBackend(QueryBackend, ABC):
    """
    Selective backends are backends that select elements from existing data.

    These can take any query as input.
    """

    opening_directive: ClassVar[Optional[Directive]] = Directive.FIND
    """
    Selecting from existing data reads as *"Find …"*.
    """

    def evaluate(self, expression: Evaluable) -> Iterable[T]:
        if isinstance(expression, Match) and expression.has_ellipsis_attributes:
            raise SelectiveBackendCannotResolveEllipsisMatch(expression)
        yield from self._evaluate(expression)

    @abstractmethod
    def _evaluate(self, expression: Evaluable) -> Iterable[T]: ...


@dataclass
class GenerativeBackend(QueryBackend, ABC):
    """
    Generative backends are backends that generate new elements.

    Generative backends have to take match expressions as input, since they need to construct new objects, and currently
    {py:class}`~krrood.entity_query_language.query.match.Match` is the only way to do so.
    """

    opening_directive: ClassVar[Optional[Directive]] = Directive.GENERATE
    """
    Generating new elements reads as *"Generate …"*.
    """

    def evaluate(self, expression: Evaluable) -> Iterable[T]:
        if not isinstance(expression, Match):
            raise GenerativeBackendQueryIsNotUnderspecifiedVariable(expression)
        yield from self._evaluate(expression)

    @abstractmethod
    def _evaluate(self, expression: Match[T]) -> Iterable[T]: ...


@dataclass
class SQLAlchemyBackend(SelectiveBackend):
    """
    A backend that selects elements from a database that is available via SQLAlchemy.
    """

    session_maker: sessionmaker
    """
    The session maker used for the database interactions.
    """

    def _evaluate(self, expression: Query) -> Iterable:
        session = self.session_maker()
        translator = eql_to_sql(expression, session)
        yield from translator.evaluate()


@dataclass
class EntityQueryLanguageBackend(SelectiveBackend):
    """
    A backend that selects elements in this python process.

    This is just ordinary EQL: each expression evaluates itself natively (queries and matches both select over their domains).
    Constructing new instances is the job of a :class:`GenerativeBackend`.
    """

    def _evaluate(self, expression: Evaluable) -> Iterable:
        yield from expression._evaluate_natively_()


@dataclass
class EntityQueryLanguageGenerativeBackend(GenerativeBackend):
    """
    A generative backend that constructs new instances deterministically: it treats a
    match's unspecified leaves as variables, enumerates every combination over their
    (discrete) domains, constructs an instance per combination via the type's
    constructor, and keeps those that satisfy the match's ``where`` conditions.
    """

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        variables: Dict[str, Variable] = {}
        for attribute_match in expression.matches_with_variables:
            self._check_attribute_match_is_suitable_for_generation(attribute_match)
            variables[attribute_match.name_from_variable_access_path] = (
                self._convert_attribute_match_to_variable(attribute_match)
            )

        expression.variable._update_domain_(
            self._generate_raw_results(expression, variables)
        )

        filtered_results = entity(expression.variable)._quantify_(
            expression._quantifier_type_
        )
        if expression._where_conditions_:
            filtered_results = filtered_results.where(*expression._where_conditions_)
        yield from filtered_results._evaluate_natively_()

    @staticmethod
    def _check_attribute_match_is_suitable_for_generation(
        attribute_match: AttributeMatch,
    ) -> None:
        """
        Raise if an assignment in the match cannot be used to generate solutions.

        :param attribute_match: The attribute match to check.
        :raises UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration: If a
            non-enum leaf is left fully unspecified (``...``), which deterministic generation
            cannot enumerate (use the :class:`ProbabilisticBackend` instead).
        """
        if isinstance(
            attribute_match.assigned_value, type(Ellipsis)
        ) and not issubclass(attribute_match.assigned_variable._type_, enum.Enum):
            raise UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration(
                attribute_match
            )

    @staticmethod
    def _convert_attribute_match_to_variable(
        attribute_match: AttributeMatch,
    ) -> Selectable:
        """
        Convert an attribute match into a variable to enumerate, handling ellipsis
        assignments for enum fields and concrete values.

        :param attribute_match: The attribute match to convert.
        :return: A variable (or symbolic expression) representing the attribute match.
        """
        if isinstance(attribute_match.assigned_value, type(Ellipsis)) and issubclass(
            attribute_match.assigned_variable._type_, enum.Enum
        ):
            return variable(
                attribute_match.assigned_variable._type_,
                list(attribute_match.assigned_variable._type_),
            )
        if isinstance(attribute_match.assigned_value, SymbolicExpression):
            return attribute_match.assigned_value
        return variable(
            type(attribute_match.assigned_value),
            [attribute_match.assigned_value],
        )

    def _generate_raw_results(
        self, expression: Match[T], variables: Dict[str, Variable]
    ) -> Iterable[T]:
        """
        Construct instances from the given match and enumerable variables.

        :param expression: The match expression to construct instances from.
        :param variables: The variables to enumerate, keyed by access- path name.
        :return: A generator yielding an instance per variable combination.
        """
        all_combinations = set_of(*variables.values())
        for combination in all_combinations._evaluate_natively_():
            for variable_name, value in zip(variables, combination.values()):
                mapped_variable = expression._get_mapped_variable_by_name(variable_name)
                mapped_variable._value_ = value
            expression._update_kwargs_from_literal_values()
            yield expression.construct_instance()


@dataclass
class ProbabilisticBackend(GenerativeBackend):
    """
    A backend that generates elements from a tractable probabilistic model using a model
    registry.
    """

    model_registry: ModelRegistry = field(default_factory=FullyFactorizedRegistry)
    """
    A model registry that can be used to resolve match statements to probabilistic
    models.
    """

    number_of_samples: int = field(kw_only=True, default=50)
    """
    The number of samples to generate.

    This is only used if the query does not specify a limit.
    """

    def _condition_and_truncate(
        self, expression: Match[T]
    ) -> tuple[UnderspecifiedParameters, ProbabilisticModel]:

        # generate parameters from example instance values
        parameters = UnderspecifiedParameters(expression)

        model = self.model_registry.get_model(parameters)

        # apply conditions from literal assignments to underspecified variables
        conditioned, _ = model.conditional(
            parameters.conditioning_assignments_from_literal_values
        )

        if conditioned is None:
            raise NoSolutionFound(expression.expression)

        # apply conditions from the where statements
        if parameters.truncation_assignments_from_where_conditions:
            truncated, _ = conditioned.truncated(
                parameters.truncation_assignments_from_where_conditions
            )
        else:
            truncated = conditioned

        # apply conditions from variable assignments to underspecified variables
        if parameters.truncation_assignments_from_krrood_variables:
            complete_event = parameters.truncation_assignments_from_krrood_variables[0]
            complete_event.fill_missing_variables(parameters.variables.values())
            for event in parameters.truncation_assignments_from_krrood_variables[1:]:
                complete_event = complete_event.intersection_with(event)
            truncated, _ = conditioned.truncated(complete_event, singleton_allowed=True)

            if truncated is None:
                raise NoSolutionFound(expression.expression)

        return parameters, truncated

    def _evaluate(self, expression: Match[T], wants_mode: bool = False) -> Iterable[T]:
        parameters, truncated = self._condition_and_truncate(expression)

        if wants_mode:
            event, _ = truncated.log_mode()
            truncated = uniform_measure_of_event(event)

            if truncated is None:
                raise NoSolutionFound(expression.expression)

        number_of_samples = expression.expression._limit_ or self.number_of_samples
        samples = truncated.sample(number_of_samples)
        log_likelihoods = truncated.log_likelihood(samples)
        samples = samples[log_likelihoods.argsort()[::-1]]

        # create new objects with the values from the samples
        for sample in samples:
            instance = parameters.construct_instance_from_model_sample(
                truncated.variables, sample
            )
            yield instance

    def evaluate_mode(self, expression: Match[T]) -> Iterable[T]:
        """
        Evaluate *expression* like :meth:`evaluate`, but draw from the circuit's own
        highest-density region instead of its whole distribution.

        Unlike :func:`~krrood.entity_query_language.factories.mode`, which aggregates
        already-materialised results and cannot appear in a query's ``where`` conditions
        (:class:`~krrood.entity_query_language.exceptions.AggregatorInWhereConditionsError`),
        this asks the underlying probabilistic circuit itself for its analytic mode. The
        query is built exactly as for a sampled evaluation; only the caller's choice of
        :meth:`evaluate` versus this method decides which is drawn.

        :param expression: The underspecified match to evaluate.
        :raises GenerativeBackendQueryIsNotUnderspecifiedVariable: If *expression* is not
            a :class:`Match`.
        :return: Instances drawn from the circuit's highest-density region, most likely
            first.
        """
        if not isinstance(expression, Match):
            raise GenerativeBackendQueryIsNotUnderspecifiedVariable(expression)
        yield from self._evaluate(expression, wants_mode=True)

    def evaluate_mode_with_log_density(
        self, expression: Match[T], fallback_sample_count: int = 1000
    ) -> Iterable[tuple[T, float]]:
        """
        Evaluate *expression* like :meth:`evaluate_mode`, but also report the answer's
        absolute log-density, so answers from differently conditioned or truncated
        evaluations of the same query shape can be ranked against one another.

        The reported log-density is the mode's own log-density inside the truncated
        model, plus the log-probability :meth:`_condition_and_truncate` accumulated
        while conditioning and truncating -- the same renormalization
        :meth:`~probabilistic_model.probabilistic_model.ProbabilisticModel.log_truncated`
        exposes, so two evaluations that truncated to differently sized regions remain
        comparable.

        Falls back to the highest-likelihood point among *fallback_sample_count* samples
        when the truncated model's exact mode is intractable
        (:class:`~probabilistic_model.exceptions.IntractableError`).

        :param expression: The underspecified match to evaluate.
        :param fallback_sample_count: Samples drawn when the exact mode cannot be
            computed.
        :raises GenerativeBackendQueryIsNotUnderspecifiedVariable: If *expression* is not
            a :class:`Match`.
        :raises NoSolutionFound: If *expression*'s evidence has no support.
        :return: One ``(instance, log_density)`` pair.
        """
        if not isinstance(expression, Match):
            raise GenerativeBackendQueryIsNotUnderspecifiedVariable(expression)

        parameters, truncated = self._condition_and_truncate(
            expression
        )

        event, mode_log_density = truncated.log_mode()
        representative_point = uniform_measure_of_event(event).sample(1)[0]
        instance = parameters.construct_instance_from_model_sample(
            truncated.variables, representative_point
        )
        yield instance, mode_log_density
