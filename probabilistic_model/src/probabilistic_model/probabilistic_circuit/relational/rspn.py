from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Type, Optional, Match, Dict, Any, Set

import pandas as pd
import rustworkx
from sortedcontainers import SortedSet

from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.feature_extractor import (
    FeatureExtractor,
    EntityCompositionDescriptor,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
)
from random_events.variable import Variable


@dataclass
class ExchangeableDistributionTemplate:

    template_distribution: RelationalProbabilisticCircuit
    latent_variables: List[Variable] = field(default_factory=list)

    def ground(
        self, parts_to_ground: List, aggregation_statistics: Dict[Variable, Any]
    ) -> ProbabilisticCircuit:
        result = ProbabilisticCircuit()
        root = ProductUnit(probabilistic_circuit=result)

        for part in parts_to_ground:
            p_part = self.template_distribution.ground(part)
            p_part.log_conditional_in_place(aggregation_statistics)
            non_latent = [v for v in p_part.variables if v not in self.latent_variables]
            p_part.marginal_in_place(non_latent)

            variable_updates = {}
            for variable in p_part.variables:
                if variable in self.latent_variables:
                    continue
                new_name = f"{part.variable}.{variable.name}"
                variable_updates[variable] = type(variable)(
                    new_name, domain=variable.domain
                )

            p_part.update_variables(variable_updates)
            if len(p_part.nodes()) == 0:
                raise ValueError("The grounding of the part failed.")
            p_part_root_index = p_part.root.index
            index_remap = result.mount(p_part.root)
            root.add_subcircuit(index_remap[p_part_root_index])

        return result


@dataclass
class RelationalProbabilisticCircuit:

    class_: Type
    """
    The class that is modelled by this distribution.
    """

    class_probabilistic_circuit: Optional[ProbabilisticCircuit] = None
    """
    The distribution of this class including its many-to-one relations.
    """

    exchangeable_distribution_templates: Dict[str, ExchangeableDistributionTemplate] = (
        field(default_factory=dict)
    )
    """
    The exchangeable distribution templates that are used to generate many-to-many relations.
    """

    specification: Optional[EntityCompositionDescriptor] = field(
        init=False, default=None
    )

    feature_extractor: Optional[FeatureExtractor] = field(init=False, default=None)

    def fit(
        self,
        instances: List[DataAccessObject],
        feature_extractor: FeatureExtractor,
        dataframe_from_parent: Optional[pd.DataFrame] = None,
    ):
        self.feature_extractor = feature_extractor
        if dataframe_from_parent is not None:
            df = dataframe_from_parent
        else:
            df = feature_extractor.create_dataframe(instances)
            df = feature_extractor.preprocess_dataframe(df)
            df = df.sort_index(axis=1)

        variables = infer_variables_from_dataframe(df)
        model = JointProbabilityTree(annotated_variables=variables)
        self.class_probabilistic_circuit = model.fit(df)
        specification = EntityCompositionDescriptor(instances[0].__class__)
        self.specification = specification
        for exchangeable_part in specification.exchangeable_parts:
            aggregations = [
                aggregation
                for aggregation, part in feature_extractor.aggregations.items()
                if part == exchangeable_part
            ]
            objects = [
                assoc.target
                for assoc in itertools.chain.from_iterable(
                    getattr(instance, exchangeable_part) for instance in instances
                )
            ]
            agg_values = [
                [
                    aggregation.apply_mapping_on_external_root(instance)
                    for aggregation in aggregations
                ]
                for instance in instances
                for _ in getattr(instance, exchangeable_part)
            ]
            df = feature_extractor.create_dataframe_for_exchangeable_parts_with_aggregations(
                objects, aggregations, agg_values
            )

            aggregation_names = [aggregation._name_ for aggregation in aggregations]

            latent_variables = [
                v.variable
                for v in infer_variables_from_dataframe(df)
                if v.variable.name in aggregation_names
            ]
            exchangeable_part_type = type(
                getattr(instances[0], exchangeable_part)[0].target
            )
            exchangeable_distribution_template = ExchangeableDistributionTemplate(
                RelationalProbabilisticCircuit(exchangeable_part_type),
                latent_variables,
            )
            child_feature_extractor = FeatureExtractor.from_instances(objects)
            exchangeable_distribution_template.template_distribution.fit(
                objects,
                child_feature_extractor,
                dataframe_from_parent=df,
            )
            self.exchangeable_distribution_templates[exchangeable_part] = (
                exchangeable_distribution_template
            )
        return self

    def ground(self, query: Match) -> ProbabilisticCircuit:
        result = self.class_probabilistic_circuit.__deepcopy__()
        queryable_object = to_dao(query.construct_instance())
        for (
            exchangeable_part_name,
            exchangeable_distribution_template,
        ) in self.exchangeable_distribution_templates.items():
            parts_to_ground = query.kwargs[exchangeable_part_name]
            feature_functions = [
                f
                for f, name in self.feature_extractor.aggregations.items()
                if name == exchangeable_part_name
            ]
            latent_by_name = {
                v.name: v for v in exchangeable_distribution_template.latent_variables
            }
            aggregation_statistics = {}
            for feature_function in feature_functions:
                name = feature_function._name_
                if name in latent_by_name:
                    aggregation_statistics[latent_by_name[name]] = (
                        feature_function.apply_mapping_on_external_root(
                            queryable_object
                        )
                    )
            result.log_conditional_in_place(aggregation_statistics)
            if len(result.nodes()) == 0:
                raise ValueError("The grounding of the class failed.")

            # find the lowest product nodes that have as scope of all the aggregations for this exchangeable part
            product_nodes_to_extend = find_lowest_product_nodes_that_model_variables(
                result,
                SortedSet(exchangeable_distribution_template.latent_variables),
            )

            grounded_exchangeable_distribution = (
                exchangeable_distribution_template.ground(
                    parts_to_ground, aggregation_statistics
                )
            )

            root_of_grounded_exchangeable_distribution = (
                grounded_exchangeable_distribution.root
            )
            node_remap = result.mount(root_of_grounded_exchangeable_distribution)

            for product_node in product_nodes_to_extend:
                product_node.add_subcircuit(
                    node_remap[root_of_grounded_exchangeable_distribution.index]
                )

            # for every distribution over the aggregation statistics in this circuit
            # calculate the probability of that aggregation statistic in the distribution and connect it to
            # the grounded template

        return result


def find_lowest_product_nodes_that_model_variables(
    circuit: ProbabilisticCircuit, variables: SortedSet[Variable]
) -> List[ProductUnit]:
    result: List[ProductUnit] = []
    ancestors_of_selected_nodes: Set[int] = set()
    for layer in reversed(circuit.layers):
        for node in layer:
            if not isinstance(node, ProductUnit):
                continue
            if node.index in ancestors_of_selected_nodes:
                continue
            if not variables.issubset(node.variables):
                continue

            result.append(node)
            ancestors_of_selected_nodes.add(node.index)
            ancestors_of_selected_nodes.update(
                rustworkx.ancestors(circuit.graph, node.index)
            )

    return result
