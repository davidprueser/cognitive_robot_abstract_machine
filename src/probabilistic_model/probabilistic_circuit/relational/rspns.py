from __future__ import annotations
from typing import Dict, Optional, Type, Iterable
from dataclasses import dataclass, field
from typing import List
from random_events.variable import Continuous, Integer, Symbolic
from typing_extensions import Any
from krrood.entity_query_language.predicate import Symbol

from probabilistic_model.distributions import (
    GaussianDistribution,
    UnivariateDistribution,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    leaf,
    ProductUnit,
)


def aggregation_statistic(func):
    func._is_aggregate_statistics = True

    return func

@dataclass
class RSPNPredicate:
    pass


@dataclass
class Adjacent(RSPNPredicate):
    nation1: Nation
    nation2: Nation


@dataclass
class Conflict(RSPNPredicate):
    nation1: Nation
    nation2: Nation


@dataclass
class Supports(RSPNPredicate):
    person: Person
    government: Government


@dataclass
class Region(Symbol):
    nations: List[Nation]
    adjacency: List[Adjacent]
    conflicts: List[Conflict]


@dataclass
class Nation(Symbol):
    government: Government
    persons: List[Person]
    supporters: List[Supports]
    gdp: float = 500

    @aggregation_statistic
    def mean_age_of_supporters(self):
        return sum(s.person.age for s in self.supporters) / len(self.supporters)


@dataclass
class Government(Symbol):
    funny: bool


@dataclass
class Person(Symbol):
    name: str
    age: float = field(default=0)


@dataclass
class DecomposedClass:
    unique_parts: List[Any] = field(default_factory=list)
    exchangeable_parts: List[Any] = field(default_factory=list)

    def copy(self):
        return DecomposedClass(
            unique_parts=self.unique_parts,
            exchangeable_parts=self.exchangeable_parts,
        )


person_part_decomposition = DecomposedClass(
    unique_parts=[],
    exchangeable_parts=[],
)

government_part_decomposition = DecomposedClass(
    unique_parts=[],
    exchangeable_parts=[],
)

nation_part_decomposition = DecomposedClass(
    unique_parts=["government"],
    exchangeable_parts=["persons"],
)

region_part_decomposition = DecomposedClass(
    unique_parts=[],
    exchangeable_parts=["nations"],
)

CLASS_SCHEMA = {
    Person: person_part_decomposition,
    Government: government_part_decomposition,
    Nation: nation_part_decomposition,
    Region: region_part_decomposition,
}

univariate_attribute_distributions = {
    "age": GaussianDistribution(Continuous("age"), 1, 1),
    "funny": GaussianDistribution(Continuous("funny"), 1, 1),
    "gdp": GaussianDistribution(Continuous("gdp"), 1, 1),
    # remove name later
    "name": GaussianDistribution(Continuous("name"), 1, 1),
}


@dataclass
class ExchangeableDistributionTemplate:
    """
    EDT is a function that returns a joint probability distribution over a set of variables.
    """

    variables: List[Continuous] = field(default_factory=list)
    """
    Random Variables that the EDT is defined over.
    """

    base_distribution: Type[UnivariateDistribution] = field(
        default_factory=GaussianDistribution
    )
    """
    The base distribution for the EDT
    """

    def __init__(
        self,
        variables: List[Continuous],
        base_distribution: Type[UnivariateDistribution],
        probabilistic_circuit,
        **kwargs,
    ):
        self.variables = variables
        self.base_distribution = base_distribution
        self.kwargs = kwargs
        self.probabilistic_circuit = probabilistic_circuit

    def __call__(self, *args, **kwargs):
        if len(self.variables) == 0:
            raise ValueError(
                "ExchangeableDistributionTemplate requires at least one variable."
            )

        # Build a product of identical univariate distributions (i.i.d.), which is exchangeable.
        product = ProductUnit(probabilistic_circuit=self.probabilistic_circuit)

        for v in self.variables:
            # Use a shared-parameter Gaussian(0, 1) for all continuous variables to ensure identical marginals.
            dist = self.base_distribution(v, **kwargs)
            unit = leaf(dist, probabilistic_circuit=self.probabilistic_circuit)
            product.add_subcircuit(unit)

        return product


@dataclass
class RSPNTemplate:
    """
    Class "C" in an RSPN consists of
    1. A set "A" of unary predicates applicable to individuals of C
    2. A vector "U_C"= (P_1,...,P_n) of unique parts.
    3. A vector "E_C"= (P_1,...,P_n) of exchangeable parts.
    4. A set of predicates (relations) of form R_1(P_1, P_2).
    5. A tractable univariate class distribution where leaves fall into one of three categories:
        5.1 L^C_A: univariate distribution over A of C
        5.2 L^C_R: EDT over predicate R involving C, E_C or U_C,
        5.3 L^C_P: Sub-SPN for part class P
    """
    instance: Any
    """
    The instance of class C to create the RSPN for.
    """

    probabilistic_circuit: Optional[ProbabilisticCircuit] = field(default=None)
    """
    The circuit this component is part of. 
    """


    attributes: List[Any] = field(default_factory=list, init=False)
    """
    Unary predicates
    """

    unique_parts: List[Any] = field(default_factory=list, init=False)
    """
    Unique parts
    """

    exchangeable_parts: List[Any] = field(default_factory=list, init=False)
    """
    Exchangeable parts
    """

    relations: List[RSPNPredicate] = field(default_factory=list, init=False)
    """
    Predicates(Relations) of form R_n(P_n, P_n)
    """

    univariate_attribute_distributions: Optional[Dict[str, UnivariateDistribution]] = field(default_factory=dict, init=False)

    """
    L^C_A: Dict of univariate distributions over all A of C
    """

    edt_over_relations: Optional[List[ExchangeableDistributionTemplate]] = field(
        default_factory=list, init=False
    )
    """
    L^C_R: List of EDTs over predicates R involving C, E_C or U_C
    """

    sub_rspns: Optional[List[Any]] = field(default_factory=list, init=False)
    """
    L^C_P: List of part classes P of C
    """

    def __post_init__(self):
        self._prepare_structure(self.instance)

    def _prepare_structure(self, instance):
        self.probabilistic_circuit = self.probabilistic_circuit or ProbabilisticCircuit()
        self.attributes = []
        self.unique_parts = []
        self.exchangeable_parts = []
        self.relations = []
        self.univariate_attribute_distributions = {}
        self.edt_over_relations = []
        self.sub_rspns = []

        schema = CLASS_SCHEMA.get(type(instance))
        if schema is None:
            raise ValueError(
                f"No schema registered for instances of type {type(instance).__name__}"
            )

        blacklist = []
        for part in schema.unique_parts:
            new_instance = getattr(instance, part)
            if isinstance(new_instance, list):
                for new_instance_part in new_instance:
                    blacklist.append(part)
                    self.unique_parts.append(RSPNTemplate(new_instance_part))
            else:
                blacklist.append(part)
                self.unique_parts.append(RSPNTemplate(new_instance))

        for part in schema.exchangeable_parts:
            new_instance = getattr(instance, part)
            if isinstance(new_instance, list):
                for new_instance_part in new_instance:
                    blacklist.append(part)
                    self.exchangeable_parts.append(RSPNTemplate(new_instance_part))
            else:
                blacklist.append(part)
                self.exchangeable_parts.append(RSPNTemplate(new_instance))

        for schema_attribute in instance.__dataclass_fields__:
            attribute = getattr(instance, schema_attribute)
            if schema_attribute in blacklist:
                continue
            else:
                if not isinstance( attribute, Iterable ) and not isinstance(attribute, RSPNPredicate):
                    self.attributes.append(schema_attribute)
                elif not isinstance(
                    getattr(instance, schema_attribute), Iterable
                ) and isinstance(getattr(instance, schema_attribute), RSPNPredicate):
                    self.relations.append(getattr(instance, schema_attribute))
                elif isinstance(getattr(instance, schema_attribute), (str, bytes)):
                    self.attributes.append(schema_attribute)
                else:
                    for value in getattr(instance, schema_attribute):
                        if not isinstance(value, RSPNPredicate):
                            raise ValueError(
                                f"Attribute {schema_attribute} must be a list of predicates (a relation) or a single predicate (an attribute), but found {value}."
                            )
                        self.relations.append(value)

        for schema_attribute in self.attributes:
            if schema_attribute in univariate_attribute_distributions:
                self.univariate_attribute_distributions[schema_attribute] = (
                    univariate_attribute_distributions[schema_attribute]
                )

        for relation in self.relations:
            fields = relation.__dataclass_fields__
            if len(fields) != 2:
                raise ValueError(
                    f"Relation {relation} must be of the form R(P1, P2) or R(C, P1) where P1, P2 are part classes of class C."
                )
            first = Continuous(list(fields.keys())[0])
            second = Continuous(list(fields.keys())[1])

            # TODO dont assume gaussian
            self.edt_over_relations.append(
                ExchangeableDistributionTemplate(
                    [first, second], GaussianDistribution, self.probabilistic_circuit
                )
            )

        self.sub_rspns = self.exchangeable_parts + self.unique_parts

    def _ground_part_classes(self, product: ProductUnit):
        for part_class in self.sub_rspns:
            if not isinstance(part_class, list):
                part_class = [part_class]
            for part in part_class:
                ground = part.ground()
                index_remap = product.probabilistic_circuit.mount(ground)
                root_index = ground.index
                product.add_subcircuit(index_remap[root_index])


    def ground(self):
        """
        Ground the given instance into a probabilistic circuit and return the root Unit
        of the grounded subcircuit representing this instance.
        """

        # Combine all components using a product unit
        product = ProductUnit(probabilistic_circuit=self.probabilistic_circuit)

        # Ground L_C^A: attributes
        for distribution in self.univariate_attribute_distributions.values():
            product.add_subcircuit(leaf(distribution, self.probabilistic_circuit))

        # Ground L_C^R: relations
        for relation_template in self.edt_over_relations:
            # TODO dont assume gaussian
            edt_product = relation_template(location=0, scale=0)
            product.add_subcircuit(edt_product)

        # Ground L_C^P: Part classes
        self._ground_part_classes(product)

        return product


