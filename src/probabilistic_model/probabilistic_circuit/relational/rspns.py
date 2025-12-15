from __future__ import annotations
import os
from dataclasses import is_dataclass
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.ormatic.dao import AlternativeMapping
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import classes_of_module
from krrood.utils import recursive_subclasses

from typing import Dict, Optional, Type, Iterable

from dataclasses import dataclass, field
from typing import List

from matplotlib import pyplot as plt
from random_events.variable import Continuous, Integer, Symbolic
from typing_extensions import Any

from krrood.entity_query_language.entity import let, an, entity
from krrood.entity_query_language.predicate import Symbol

from probabilistic_model.distributions import (
    GaussianDistribution,
    UnivariateDistribution,
    BernoulliDistribution,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    leaf,
    ProductUnit,
)

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
    Region: region_part_decomposition
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

    base_distribution: Type[UnivariateDistribution] = field(default_factory=GaussianDistribution)
    """
    The base distribution for the EDT
    """

    def __init__(self, variables: List[Continuous], base_distribution: Type[UnivariateDistribution], probabilistic_circuit, **kwargs):
        self.variables = variables
        self.base_distribution = base_distribution
        self.kwargs = kwargs
        self.probabilistic_circuit = probabilistic_circuit


    def __call__(self, *args, **kwargs):
        if len(self.variables) == 0:
            raise ValueError("ExchangeableDistributionTemplate requires at least one variable.")

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

    probabilistic_circuit: ProbabilisticCircuit

    attributes: List[Any] = field(default_factory=list)
    """
    Unary predicates
    """

    unique_parts: List[Any] = field(default_factory=list)
    """
    Unique parts
    """

    exchangeable_parts: List[Any] = field(default_factory=list)
    """
    Exchangeable parts
    """

    relations: List[RSPNPredicate] = field(default_factory=list)
    """
    Predicates(Relations) of form R_n(P_n, P_n)
    """

    univariate_attribute_distributions: Optional[Dict[str, UnivariateDistribution]] = field(default_factory=dict)
    """
    L^C_A: Dict of univariate distributions over all A of C
    """

    edt_over_relations: Optional[List[ExchangeableDistributionTemplate]] = field(default_factory=list)
    """
    L^C_R: List of EDTs over predicates R involving C, E_C or U_C
    """

    sub_rspns: Optional[List[Any]] = None
    """
    L^C_P: List of part classes P of C
    """

    def _ground_part_classes(self, product: ProductUnit, instance: Any):
        for part_class_name in self.sub_rspns:
            part_class = getattr(instance, part_class_name)
            if not isinstance(part_class, list):
                part_class = [part_class]
            for part in part_class:
                product.add_subcircuit(self.ground(part))

    def _prepare_structure(self, instance):
        # reset all class attributes
        self.attributes = []
        self.unique_parts = []
        self.exchangeable_parts = []
        self.relations = []
        self.univariate_attribute_distributions = {}
        self.edt_over_relations = []
        self.sub_rspns = []

        schema = CLASS_SCHEMA.get(type(instance))
        if schema is None:
            raise ValueError(f"No schema registered for instances of type {type(instance).__name__}")

        self.unique_parts = schema.unique_parts

        self.exchangeable_parts = schema.exchangeable_parts

        for attribute in instance.__dataclass_fields__:
            if attribute in self.unique_parts or attribute in self.exchangeable_parts:
                continue
            else:
                if not isinstance(getattr(instance, attribute), Iterable) and not isinstance(getattr(instance, attribute), RSPNPredicate):
                    self.attributes.append(attribute)
                elif not isinstance(getattr(instance, attribute), Iterable) and isinstance(getattr(instance, attribute), RSPNPredicate):
                    self.relations.append(getattr(instance, attribute))
                elif isinstance(getattr(instance, attribute), (str, bytes)):
                    self.attributes.append(attribute)
                else:
                    for value in getattr(instance, attribute):
                        if not isinstance(value, RSPNPredicate):
                            raise ValueError(f"Attribute {attribute} must be a list of predicates (a relation) or a single predicate (an attribute), but found {value}.")
                        self.relations.append(value)

        for attribute in self.attributes:
            if attribute in univariate_attribute_distributions:
                self.univariate_attribute_distributions[attribute] = univariate_attribute_distributions[attribute]

        for relation in self.relations:
            fields = relation.__dataclass_fields__
            if len(fields) != 2:
                raise ValueError(f"Relation {relation} must be of the form R(P1, P2) or R(C, P1) where P1, P2 are part classes of class C.")
            first = Continuous(list(fields.keys())[0])
            second = Continuous(list(fields.keys())[1])

            #TODO dont assume gaussian
            self.edt_over_relations.append(ExchangeableDistributionTemplate([first, second], GaussianDistribution, self.probabilistic_circuit))

        self.sub_rspns = self.exchangeable_parts + self.unique_parts


    def ground(self, instance):
        """
        Ground the given instance into a probabilistic circuit and return the root Unit
        of the grounded subcircuit representing this instance.
        """

        self._prepare_structure(instance)

        # Combine all components using a product unit
        product = ProductUnit(probabilistic_circuit=self.probabilistic_circuit)

        # Ground L_C^A: attributes
        for distribution in self.univariate_attribute_distributions.values():
            product.add_subcircuit(leaf(distribution, self.probabilistic_circuit))

        # Ground L_C^R: relations
        for relation_template in self.edt_over_relations:
            #TODO dont assume gaussian
            edt_product = relation_template(location=0, scale=0)
            product.add_subcircuit(edt_product)

        # Ground L_C^P: Part classes
        self._ground_part_classes(product, instance)

        return product

def example():
    david = Person("David", 25)
    tom = Person("Tom", 27)
    checker_chan = Person("Simon Wallukat", 28)

    cdu = Government(funny=True)
    persons = [david, tom, checker_chan]
    supporters = [Supports(checker_chan, cdu)]
    n1 = Nation(government=cdu, supporters=supporters, persons=persons)

    daniel = Person("Daniel", 50)
    tede = Person("Tede", 25)
    daniel_union = Government(funny=False)

    knowrob_supporters = [daniel, tede]

    knowrob_nation = Nation(
        government=daniel_union,
        persons=[daniel, tede],
        supporters=[Supports(daniel, daniel_union), Supports(tede, daniel_union)],
        gdp=-10,
    )

    region = Region(
        nations=[n1, knowrob_nation],
        adjacency=[Adjacent(n1, knowrob_nation)],
        conflicts=[Conflict(n1, knowrob_nation)],
    )

    n2 = let(Nation, [])
    p3 = let(Person, [])
    q = an(
        entity(
            n2,
            n2.government.funny == True,
            n2.persons == [david, tom, p3],
            )
    )
    rspn = RSPNTemplate(ProbabilisticCircuit())
    ground_region = rspn.ground(region)
    # pretty_print_ground(ground_region)
    ground_region.probabilistic_circuit.plot_structure()
    plt.show()
    print(ground_region.is_decomposable())
