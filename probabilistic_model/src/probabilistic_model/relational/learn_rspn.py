from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from random_events.variable import Continuous, Integer

from probabilistic_model.distributions import (
    BernoulliDistribution,
    GaussianDistribution,
    UnivariateDistribution,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)

from probabilistic_model.probabilistic_circuit.relational.rspns import (
    CLASS_SCHEMA,
    DecomposedClass,
    RSPNPredicate,
    ExchangeableDistributionTemplate,
    RSPNTemplate,
)
from ...learning.jpt.jpt import JPT
from ...learning.jpt.variables import infer_variables_from_dataframe


def _is_part(attribute, part_decomposition) -> bool:
    unique_parts_keys = [x for x, _ in part_decomposition.unique_parts]
    exchangeable_parts_keys = [x for x, _ in part_decomposition.exchangeable_parts]
    return attribute in unique_parts_keys or attribute in exchangeable_parts_keys


# def _learn_child(C_child: Type, T: Sequence[Any], cfg: LearnConfig) -> ProductUnit | SumUnit:
#     # Recursively learn for child class using its full scope V_child
#     schema = CLASS_SCHEMA.get(C_child)
#     if schema is None:
#         raise ValueError(f"No schema registered for class {C_child.__name__}")
#     V_child: List[str] = list(schema.attributes) + list(schema.unique_parts) + list(schema.exchangeable_parts)
#     return LearnRSPN(C_child, T, V_child, cfg)


def _is_attribute(attribute, part_decomp) -> bool:
    return not _is_part(attribute, part_decomp)


def is_aggregate_statistics(func):
    return getattr(func, "_is_aggregate_statistics", False)


def get_aggregate_statistics(instance):
    statistics = []
    for name in dir(instance):
        if name.startswith("__"):
            continue
        try:
            attr = getattr(instance, name)
        except Exception:
            continue
        if callable(attr) and is_aggregate_statistics(attr):
            statistics.append(attr())
            continue

    return statistics


def LearnRSPN(cls, instances, class_spec) -> RSPNTemplate:
    """
    Learn an RSPN for class C from instances T over variables V, implementing Algorithm 1.

    - Attributes become univariate leaves (Gaussian for numeric, Bernoulli for boolean)
    - Relation aggregates become Bernoulli leaves over presence (1 if present, else 0)
    - Parts recurse into their class (unique part: map one-to-one; exchangeable part: flatten list)
    - Independent partitions become product nodes; clustering on instances becomes sum nodes with weights

    Returns the root node (ProductUnit or SumUnit) within a ProbabilisticCircuit.
    """
    df_data: Dict[str, List[float]] = {}
    if not isinstance(instances, list):
        instances = [instances]

    attribute_values = {}
    relation_values = {}
    unique_values = {}
    exchangeable_values = {}

    for instance in instances:
        aggregation_values = get_aggregate_statistics(instance)
        assert isinstance(instance, cls)

        if len(class_spec["attributes"]) > 0:
            for attribute in class_spec["attributes"]:
                if attribute not in df_data:
                    df_data[attribute] = [getattr(instance, attribute)]
                else:
                    # attribute_values[attribute] += getattr(instance, attribute)
                    df_data[attribute].append(getattr(instance, attribute))

        if len(class_spec["relations"]) > 0:
            for relation in class_spec["relations"]:
                # relation_values += [value for value, type in aggregation_values if type == relation]
                if relation not in df_data:
                    df_data[relation] = [
                        value for value, type in aggregation_values if type == relation
                    ]
                else:
                    df_data[relation] += [
                        value for value, type in aggregation_values if type == relation
                    ]

        if len(class_spec["unique_parts"]) > 0:
            for unique_part in class_spec["unique_parts"]:
                # unique_values += [value for value, type in aggregation_values if type == unique_part]
                if unique_part not in df_data:
                    df_data[unique_part] = [
                        value
                        for value, type in aggregation_values
                        if type == unique_part
                    ]
                else:
                    df_data[unique_part] += [
                        value
                        for value, type in aggregation_values
                        if type == unique_part
                    ]

        if len(class_spec["exchangeable_parts"]) > 0:
            for exchangeable_part in class_spec["exchangeable_parts"]:
                # exchangeable_values += [value for value, type in aggregation_values if type == exchangeable_part]
                if exchangeable_part not in df_data:
                    df_data[exchangeable_part] = [
                        value
                        for value, type in aggregation_values
                        if type == exchangeable_part
                    ]
                else:
                    df_data[exchangeable_part] += [
                        value
                        for value, type in aggregation_values
                        if type == exchangeable_part
                    ]
                x = 0

    df = pd.DataFrame(df_data)
    variables = infer_variables_from_dataframe(df)
    jpt = JPT(variables)
    jpt = jpt.fit(df)
    rspn = RSPNTemplate(class_spec, jpt)
    # jpt.plot_structure()
    # plt.show()
    return rspn

    # # Collect a flat table (df_data) of attributes and aggregation statistics
    # df_data: Dict[str, List[float]] = {}
    #
    # # for instance in instance:
    # fields = getattr(instance, "__dataclass_fields__", {})
    # for attribute in fields.keys():
    #     if _is_attribute(attribute, cls):
    #         attribute_value = getattr(instance, attribute)
    #         if isinstance(attribute_value, list):
    #             assert len(attribute_value) >= 0
    #             values = get_aggregate_statistics(instance)
    #             for value, type in values:
    #                 if type == attribute:
    #                     df_data[attribute] = [value]
    #             # df_data[attribute] = values
    #             continue
    #         else:
    #             if isinstance(attribute_value, bool):
    #                 values = float(getattr(instance, attribute))
    #             else:
    #                 values = getattr(instance, attribute)
    #             df_data[attribute] = [values]
    #             continue
    #
    #     elif _is_part(attribute, cls):
    #         continue
