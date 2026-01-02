from __future__ import annotations
from typing import Dict, Optional, Type, Iterable
from dataclasses import dataclass, field
from typing import List
import copy
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
    Unit, SumUnit,
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
    name: str = "World"

    @aggregation_statistic
    def adjacency_density(self):
        return len(self.adjacency) / len(self.nations), "adjacency"

    @aggregation_statistic
    def conflict_density(self):
        return len(self.conflicts) / len(self.nations), "conflicts"


@dataclass
class Nation(Symbol):
    government: Government = None
    persons: List[Person] = field(default_factory=list)
    supporters: List[Supports] = field(default_factory=list)
    gdp: float = 500
    name: str = "Germany"

    @aggregation_statistic
    def mean_age_of_supporters(self):
        return sum(s.person.age for s in self.supporters) / len(self.supporters), "supporters"


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
    unique_parts=[("government", Government)],
    exchangeable_parts=[("persons", Person)],
)

region_part_decomposition = DecomposedClass(
    unique_parts=[],
    exchangeable_parts=[("nations", Nation)],
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
        **kwargs,
    ):
        self.variables = variables
        self.base_distribution = base_distribution
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        if len(self.variables) == 0:
            raise ValueError(
                "ExchangeableDistributionTemplate requires at least one variable."
            )

        circuit = ProbabilisticCircuit()
        product = ProductUnit(probabilistic_circuit=circuit)

        for v in self.variables:
            dist = self.base_distribution(v, **kwargs)
            unit = leaf(dist, probabilistic_circuit=circuit)
            product.add_subcircuit(unit)

        return product

class_spec_nation = {
    "exchangeable_parts": [("persons", Person)],
    "unique_parts": [("government", Government)],
    "attributes": ["gdp"],
    "relations": [Supports]
}
class_spec_gov = {
    "attributes": ["funny"],
    "relations": [],
    "exchangeable_parts": [],
    "unique_parts": []

}
class_spec_region = {
    "exchangeable_parts": [("nations", Nation)],
    "unique_parts": [],
    "relations": [Adjacent, Conflict],
    "attributes": []
}
class_spec_person = {
    "attributes": ["age", "name"],
    "relations": [],
    "exchangeable_parts": [],
    "unique_parts": []
}

classes = {
    Nation: class_spec_nation,
    Government: class_spec_gov,
    Region: class_spec_region,
    Person: class_spec_person
}


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

    class_spec: Dict[str, List] = field(init=True)

    probabilistic_circuit: Optional[ProbabilisticCircuit] = field(default=None, init=False)
    """
    The circuit this component is part of. 
    """

    # attributes: Optional[List[Any]] = field(default_factory=list)
    # """
    # Unary predicates
    # """
    #
    # unique_parts: Optional[List[Any]] = field(default_factory=list)
    # """
    # Unique parts
    # """
    #
    # exchangeable_parts: Optional[List[Any]] = field(default_factory=list)
    # """
    # Exchangeable parts
    # """
    #
    # relations: Optional[List[Any]] = field(default_factory=list)
    # """
    # Predicates(Relations) of form R_n(P_n, P_n)
    # """

    univariate_attribute_distributions: Optional[Dict[str, UnivariateDistribution]] = field(default_factory=dict)
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
        self.attributes = self.class_spec["attributes"]
        self.unique_parts = []
        for part_name, unique_part in self.class_spec["unique_parts"]:
            self.unique_parts.append((part_name, RSPNTemplate(classes[unique_part])))

        self.exchangeable_parts = []
        for part_name, exchangeable_part in self.class_spec["exchangeable_parts"]:
            self.exchangeable_parts.append((part_name, RSPNTemplate(classes[exchangeable_part])))

        self.relations = self.class_spec["relations"]
        self._prepare_structure()

    def _prepare_structure(self):
        self.probabilistic_circuit = self.probabilistic_circuit or ProbabilisticCircuit()
        self.univariate_attribute_distributions = {}
        self.edt_over_relations = []
        self.sub_rspns = []
        self.edt_products = []

        product = ProductUnit(probabilistic_circuit=self.probabilistic_circuit)
        self.fix_attribute_distributions(product)
        self.fix_edt_over_relations(product)
        self.fix_sub_spns(product)

    def fix_attribute_distributions(self, product):
        for schema_attribute in self.attributes:
            if schema_attribute in univariate_attribute_distributions:
                self.univariate_attribute_distributions[schema_attribute] = (
                    univariate_attribute_distributions[schema_attribute]
                )

        for distribution in self.univariate_attribute_distributions.values():
            product.add_subcircuit(leaf(distribution, self.probabilistic_circuit))

    def fix_edt_over_relations(self, product):
        for relation in self.relations:
            fields = relation.__dataclass_fields__
            if len(fields) != 2:
                raise ValueError(
                    f"Relation {relation} must be of the form R(P1, P2) or R(C, P1) where P1, P2 are part classes of class C."
                )
            first = Continuous(list(fields.keys())[0])
            second = Continuous(list(fields.keys())[1])

            self.edt_over_relations.append(
                ExchangeableDistributionTemplate(
                    [first, second], GaussianDistribution
                )
            )
        for relation_template in self.edt_over_relations:
            # TODO dont assume gaussian
            edt_product = relation_template(location=0, scale=0)
            self.edt_products.append(edt_product)
            index_remap = product.probabilistic_circuit.mount(edt_product)
            root_index = edt_product.index
            product.add_subcircuit(index_remap[root_index])

    def fix_sub_spns(self, product):
        self.sub_rspns = [p[1] for p in self.exchangeable_parts + self.unique_parts]

        for part_class in self.sub_rspns:
            index_remap = product.probabilistic_circuit.mount(part_class.probabilistic_circuit.root)
            root_index = part_class.probabilistic_circuit.root.index
            product.add_subcircuit(index_remap[root_index])

    def ground(self, instance: Any) -> Unit:
        """
        Ground the RSPN template for a specific instance.
        :param instance: The object to ground the template for.
        :return: A grounded SPN as a Unit.
        """
        # Create a product unit for this grounding
        grounded_product = ProductUnit(probabilistic_circuit=ProbabilisticCircuit())

        instance_repr = getattr(instance, "name", str(id(instance)))

        # 1. Ground attributes (L^C_A)
        for attr_name, distribution in self.univariate_attribute_distributions.items():
            # Create a grounded variable name, e.g., "age(O)"
            grounded_var_name = f"{attr_name}({instance_repr})"

            # Create a new variable of the same type
            original_var = distribution.variable
            grounded_var = original_var.__class__(grounded_var_name)

            # Create a grounded distribution
            import copy
            grounded_dist = copy.deepcopy(distribution)
            grounded_dist.variable = grounded_var

            grounded_product.add_subcircuit(leaf(grounded_dist, grounded_product.probabilistic_circuit))

        # 2. Ground relations (L^C_R)
        for relation, relation_template in zip(self.relations, self.edt_over_relations):
            # Instantiate the EDT for this instance.
            grounded_edt = relation_template(location=0, scale=1)

            # Ground the variables in the EDT product
            for l in grounded_edt.leaves:
                orig_var = l.distribution.variable
                new_var_name = f"{orig_var.name}({instance_repr})"
                l.distribution.variable = orig_var.__class__(new_var_name)

            # Mount the grounded_edt into our current circuit
            # We use the fact that grounded_edt already has a circuit from relation_template call.
            index_remap = grounded_product.probabilistic_circuit.mount(grounded_edt)
            grounded_product.add_subcircuit(index_remap[grounded_edt.index])

        # 3. Ground parts (L^C_P)
        # Unique parts
        for part_name, part_template in self.unique_parts:
            part_instance = getattr(instance, part_name, None)
            if part_instance is not None:
                grounded_part = part_template.ground(part_instance)
                # Mount the grounded part into our current circuit
                index_remap = grounded_product.probabilistic_circuit.mount(grounded_part)
                grounded_product.add_subcircuit(index_remap[grounded_part.index])

        # Exchangeable parts
        for part_name, part_template in self.exchangeable_parts:
            part_instances = getattr(instance, part_name, [])
            for p_inst in part_instances:
                grounded_part = part_template.ground(p_inst)
                # Mount the grounded part into our current circuit
                index_remap = grounded_product.probabilistic_circuit.mount(grounded_part)
                grounded_product.add_subcircuit(index_remap[grounded_part.index])

        return grounded_product



