from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Type, Optional, Match, Dict, Any, Set

import pandas as pd
import rustworkx
from sortedcontainers import SortedSet

from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.feature_extraction.feature_extractor import (
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
            conditioning_result, _ = p_part.log_conditional_in_place(
                aggregation_statistics
            )
            if conditioning_result is None:
                # Impossible conditioning (e.g. unresolved query attributes produced
                # out-of-distribution statistic values); fall back to unconditioned part.
                p_part = self.template_distribution.ground(part)
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
        dataframe_from_parent: Optional[pd.DataFrame] = None,
    ):
        self.feature_extractor = FeatureExtractor.from_instances(instances)
        if dataframe_from_parent is not None:
            df = dataframe_from_parent
        else:
            df = self.feature_extractor.create_dataframe(instances)
            df = self.feature_extractor.preprocess_dataframe(df)
            df = df.sort_index(axis=1)

        variables = infer_variables_from_dataframe(df)
        model = JointProbabilityTree(annotated_variables=variables)
        self.class_probabilistic_circuit = model.fit(df)
        specification = EntityCompositionDescriptor(instances[0].__class__)
        self.specification = specification
        for exchangeable_part in specification.exchangeable_parts:
            aggregations = self.feature_extractor.exchangeable_features[
                exchangeable_part
            ]
            agg_indices = [
                next(
                    i for i, f in enumerate(self.feature_extractor.features) if f is agg
                )
                for agg in aggregations
            ]
            agg_names = [agg._name_ for agg in aggregations]
            objects = [
                assoc.target
                for assoc in itertools.chain.from_iterable(
                    getattr(instance, exchangeable_part) for instance in instances
                )
            ]
            first_child_type = type(getattr(instances[0], exchangeable_part)[0].target)
            child_attr_names = [
                col.key
                for col in EntityCompositionDescriptor(first_child_type).attributes
            ]
            rows = []
            for instance in instances:
                mapped = self.feature_extractor.apply_mapping(instance)
                agg_row = [mapped[i] for i in agg_indices]
                for assoc in getattr(instance, exchangeable_part):
                    rows.append(
                        agg_row
                        + [getattr(assoc.target, name) for name in child_attr_names]
                    )
            df = pd.DataFrame(columns=agg_names + child_attr_names, data=rows)
            aggregation_names = agg_names

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
            exchangeable_distribution_template.template_distribution.fit(
                objects,
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
            feature_functions = self.feature_extractor.exchangeable_features[
                exchangeable_part_name
            ]
            latent_by_name = {
                v.name: v for v in exchangeable_distribution_template.latent_variables
            }
            aggregation_statistics = {}
            for feature_function in feature_functions:
                name = feature_function._name_
                if name in latent_by_name:
                    aggregation_instance = (
                        queryable_object.from_dao().get_aggregation_class_by_part_name(
                            exchangeable_part_name
                        )
                    )
                    value = feature_function.apply_mapping_on_external_root(
                        aggregation_instance,
                    )
                    latent_var = latent_by_name[name]
                    # Only condition on statistics whose value is in the training domain;
                    # unresolved query attributes (e.g. type=...) can produce impossible
                    # values (e.g. chair_count=0 when the training data has no such rooms)
                    # that would collapse the circuit.
                    try:
                        latent_var.make_value(value)
                        aggregation_statistics[latent_var] = value
                    except (ValueError, TypeError):
                        pass
            # find the lowest product nodes BEFORE conditioning — conditioning removes the latent
            # variables from the circuit scope, so they cannot be found afterwards
            product_nodes_to_extend = find_lowest_product_nodes_that_model_variables(
                result,
                SortedSet(exchangeable_distribution_template.latent_variables),
            )

            conditioning_result, _ = result.log_conditional_in_place(
                aggregation_statistics
            )
            if conditioning_result is None:
                # Impossible conditioning — unresolved query attributes produced
                # out-of-distribution statistic values. Restore the unconditioned circuit
                # so grounding can still produce a valid (if less specific) distribution.
                result = self.class_probabilistic_circuit.__deepcopy__()
                product_nodes_to_extend = (
                    find_lowest_product_nodes_that_model_variables(
                        result,
                        SortedSet(exchangeable_distribution_template.latent_variables),
                    )
                )
            if len(result.nodes()) == 0:
                raise ValueError("The grounding of the class failed.")

            # conditioning may have pruned some product nodes; keep only surviving ones
            product_nodes_to_extend = [
                n for n in product_nodes_to_extend if n.index is not None
            ]

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
