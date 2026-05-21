from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Type, Optional

from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.parametrization.feature_extractor import (
    FeatureExtractor,
    RelationalSumProductNetworkSpecification,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)


@dataclass
class ExchangeableDistributionTemplate:

    template_distribution: RelationalProbabilisticCircuit

    def fit(
        self, instances: List[DataAccessObject], feature_extractor: FeatureExtractor
    ):
        pass


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

    exchangeable_distribution_templates: List[ExchangeableDistributionTemplate] = field(
        default_factory=list
    )
    """
    The exchangeable distribution templates that are used to generate many-to-many relations.
    """

    def fit(
        self, instances: List[DataAccessObject], feature_extractor: FeatureExtractor
    ):
        df = feature_extractor.create_dataframe(instances)
        df = feature_extractor.preprocess_dataframe(df)
        df = df.sort_index(axis=1)

        variables = infer_variables_from_dataframe(df)
        model = JointProbabilityTree(annotated_variables=variables)
        self.class_probabilistic_circuit = model.fit(df)
        specification = RelationalSumProductNetworkSpecification(instances[0].__class__)
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
            x = 0

        # for every exchangeable part
        # collect all children from all instances
        # enrich them with their aggregation statistics from this instances
        # child = RelationalProbabilisticCircuit(relationship target class).fit(the data you collected above)
        # self.exchangeable_distribution_templates.append(ExchangeableDistributionTemplate(child)
        # create an EDT with the fitted rspn as template_distribution (RECURSIVE!!!!!!!!!!!!!!!!)

        return
