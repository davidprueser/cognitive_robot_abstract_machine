from __future__ import annotations
from typing import Dict

from dataclasses import dataclass, field
from typing import List

from matplotlib import pyplot as plt
from random_events.variable import Continuous, Integer
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
    attributes: List[Any] = field(default_factory=list)
    unique_parts: List[Any] = field(default_factory=list)
    exchangeable_parts: List[Any] = field(default_factory=list)


    def copy(self):
        return DecomposedClass(
            attributes=self.attributes,
            unique_parts=self.unique_parts,
            exchangeable_parts=self.exchangeable_parts,
        )

person_part_decomposition = DecomposedClass(
    attributes=["age"],
    unique_parts=[],
    exchangeable_parts=[],
)

government_part_decomposition = DecomposedClass(
    attributes=["funny"],
    unique_parts=[],
    exchangeable_parts=[],
)

nation_part_decomposition = DecomposedClass(
    attributes=["gdp", "supporters"],
    unique_parts=["government"],
    exchangeable_parts=["persons"],
)

region_part_decomposition = DecomposedClass(
    attributes=["adjacency", "conflicts"],
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
}


@dataclass
class RSPNTemplate:

    probabilistic_circuit: ProbabilisticCircuit

    # For each attribute name provide a univariate distribution template to use during grounding
    univariate_attribute_distributions: Dict[str, UnivariateDistribution] = field(
        default_factory=lambda: univariate_attribute_distributions.copy()
    )

    # Optional per-relation-type Bernoulli parameter p. Defaults to 0.5 if not specified.
    relation_bernoulli_p: Dict[type, float] = field(default_factory=dict)

    def _clone_univariate(self, instance: Any, attribute: str, template: UnivariateDistribution) -> UnivariateDistribution:
        """
        Create a fresh univariate distribution for the given attribute based on a template.
        Ensures decomposability by assigning a unique variable per object-instance and attribute.
        Currently supports Gaussian explicitly; otherwise falls back to a standard normal.
        """
        # Unique variable name per instance-attribute to guarantee disjoint scopes in products
        var_name = f"{type(instance).__name__}.{attribute}:{id(instance)}"
        variable = Continuous(name=var_name)
        if isinstance(template, GaussianDistribution):
            return GaussianDistribution(variable, template.location, template.scale)
        # Safe minimal default
        return GaussianDistribution(variable, 0.0, 1.0)

    def _clone_relation_distribution(self, instance: RSPNPredicate) -> UnivariateDistribution:
        """
        Create a fresh Bernoulli distribution leaf for a relationship instance.
        Uses BernoulliDistribution over {0,1} with a unique variable name per instance.
        """
        # Create a new variable name including the instance id to avoid name clashes
        var_name = f"{type(instance).__name__}:{id(instance)}"
        p = self.relation_bernoulli_p.get(type(instance), 0.5)
        return BernoulliDistribution(Integer(var_name), p=p)

    def _predicate_leaf(self, instance: RSPNPredicate):
        distribution = self._clone_relation_distribution(instance)
        return leaf(distribution, probabilistic_circuit=self.probabilistic_circuit)

    def _relationship_leaves_from_value(self, value):
        """
        If the provided value encodes relationships (predicate or list of predicates),
        return a list of leaf units for them; otherwise return an empty list.
        """
        # Single predicate
        if isinstance(value, RSPNPredicate):
            return [self._predicate_leaf(value)]

        # List of predicates
        if isinstance(value, list):
            leaves = []
            for v in value:
                if isinstance(v, RSPNPredicate):
                    leaves.append(self._predicate_leaf(v))
            return leaves

        return []

    def _ground_attribute(self, instance, attribute: str):
        """
        Ground a single attribute. If it's a relationship (or list of), create the
        corresponding relation leaves. Otherwise, if a univariate template exists,
        create a standard univariate leaf. Returns a list of units (may be empty).
        """
        if not hasattr(instance, attribute):
            return []

        value = getattr(instance, attribute)

        # Relationship handling takes precedence
        rel_leaves = self._relationship_leaves_from_value(value)
        if rel_leaves:
            return rel_leaves

        template = self.univariate_attribute_distributions.get(attribute)
        if template is None:
            return []

        distribution = self._clone_univariate(instance, attribute, template)
        return [leaf(distribution, probabilistic_circuit=self.probabilistic_circuit)]

    def _ground_attributes(self, instance, schema: DecomposedClass, product: ProductUnit):
        for attribute in schema.attributes:
            for unit in self._ground_attribute(instance, attribute):
                product.add_subcircuit(unit)

    def _ground_unique_parts(self, instance, schema: DecomposedClass, product: ProductUnit):
        for up_field in schema.unique_parts:
            if not hasattr(instance, up_field):
                continue
            child = getattr(instance, up_field)
            if child is None:
                continue
            product.add_subcircuit(self.ground(child))

    def _ground_exchangeable_parts(self, instance, schema: DecomposedClass, product: ProductUnit):
        for ep_field in schema.exchangeable_parts:
            if not hasattr(instance, ep_field):
                continue
            children = getattr(instance, ep_field) or []
            if not isinstance(children, list):
                children = [children]
            for child in children:
                product.add_subcircuit(self.ground(child))

    def ground(self, instance):
        """
        Ground the given instance into a probabilistic circuit and return the root Unit
        of the grounded subcircuit representing this instance.
        """
        # Relationship instances are modeled as binary attributes (leaves)
        if isinstance(instance, RSPNPredicate):
            return self._predicate_leaf(instance)

        schema = CLASS_SCHEMA.get(type(instance))
        if schema is None:
            raise ValueError(f"No schema registered for instances of type {type(instance).__name__}")

        # Combine all components using a product unit
        product = ProductUnit(probabilistic_circuit=self.probabilistic_circuit)

        # Attributes as leaves (univariate or relation-derived)
        self._ground_attributes(instance, schema, product)

        # Unique parts: a single object per field
        self._ground_unique_parts(instance, schema, product)

        # Exchangeable parts: a list of objects per field
        self._ground_exchangeable_parts(instance, schema, product)

        return product


if __name__ == "__main__":  # quick demo if this file is run
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
