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

from probabilistic_model.probabilistic_circuit.relational.rspns import CLASS_SCHEMA, DecomposedClass, RSPNPredicate, \
    ExchangeableDistributionTemplate, RSPNTemplate
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



def LearnRSPN(part_decompositions, instance) -> RSPNTemplate:
    """
    Learn an RSPN for class C from instances T over variables V, implementing Algorithm 1.

    - Attributes become univariate leaves (Gaussian for numeric, Bernoulli for boolean)
    - Relation aggregates become Bernoulli leaves over presence (1 if present, else 0)
    - Parts recurse into their class (unique part: map one-to-one; exchangeable part: flatten list)
    - Independent partitions become product nodes; clustering on instances becomes sum nodes with weights

    Returns the root node (ProductUnit or SumUnit) within a ProbabilisticCircuit.
    """

    # Collect a flat table (df_data) of attributes and aggregation statistics
    df_data: Dict[str, List[float]] = {}

    # for instance in instance:
    fields = getattr(instance, "__dataclass_fields__", {})
    for attribute in fields.keys():
        if _is_attribute(attribute, part_decompositions):
            attribute_value = getattr(instance, attribute)
            if isinstance(attribute_value, list):
                assert len(attribute_value) >= 0
                values = get_aggregate_statistics(instance)
                for value, type in values:
                    if type == attribute:
                        df_data[attribute] = [value]
                # df_data[attribute] = values
                continue
            else:
                if isinstance(attribute_value, bool):
                    values = float(getattr(instance, attribute))
                else:
                    values = getattr(instance, attribute)
                df_data[attribute] = [values]
                continue

        elif _is_part(attribute, part_decompositions):
            continue

    df = pd.DataFrame(df_data)
    print(df)
    variables = infer_variables_from_dataframe(df)
    jpt = JPT(variables)
    jpt = jpt.fit(df)
    rspn = RSPNTemplate(jpt)
    # jpt.plot_structure()
    # plt.show()
    return jpt

    #     # 2) Aggregation statistics (methods decorated with @aggregation_statistic)
    #     # We iterate over bound attributes and call methods marked as aggregate statistics
    #     for name in dir(instance):
    #         if name.startswith("__"):
    #             continue
    #         try:
    #             attr = getattr(instance, name)
    #         except Exception:
    #             continue
    #         if callable(attr) and is_aggregate_statistics(attr):
    #             try:
    #                 val = attr()
    #             except Exception:
    #                 continue
    #             _append_value(name, val)
    #
    # # Align column lengths: ensure every column has a value per instance (use NaN for missing)
    # # This keeps DataFrame construction robust when some attributes are missing in some instances.
    # num_instances = len(instances)
    # for k, col in df_data.items():
    #     if len(col) < num_instances:
    #         # pad with NaNs to match number of instances
    #         col.extend([float("nan")] * (num_instances - len(col)))
    #
    # # Build DataFrame for potential downstream learning steps
    # if df_data:
    #     df = pd.DataFrame(df_data)
    #     # Optionally infer variables and fit a simple JPT (kept here for future use)
    #     try:
    #         variables = infer_variables_from_dataframe(df)
    #         jpt = JPT(variables)  # Construct; fitting can be added later as needed
    #     except Exception:
    #         # Don't fail learning just because JPT inference isn't applicable yet
    #         pass
    #
    # # For now, return the empty product unit; subsequent steps can add leaves based on df_data
    # jpt.fit(df)
    # # jpt.plot_structure()
    # # plt.show()
    # return prod
    #
    #

    # V = list(V)
    # if len(V) == 0:
    #     return ProductUnit(probabilistic_circuit=pc)
    #
    # df_data = {}
    # for v in V:
    #     df_data[v] = _collect_numeric(T, v)
    #
    #
    # gdps = [nation.gdp for nation in T]
    # supporters = [nation.mean_age_of_supporters() for nation in T]
    # df = pd.DataFrame({
    #     "gdps": gdps,
    #     "supporters": supporters
    # })
    # variables = infer_variables_from_dataframe(df)
    # jpt = JPT(variables)
    # jpt = jpt.fit(df)
    # jpt.plot_structure()
    # plt.show()




    # if len(V) == 1:
    #     v = V[0]
    #     # schema: Optional[DecomposedClass] = CLASS_SCHEMA.get(C)
    #     # if schema is None:
    #     #     raise ValueError(f"No schema registered for class {C.__name__}")
    #     if _is_attribute(C, v):
    #         prod.add_subcircuit(leaf(_estimate_univariate(C, T, v, cfg), probabilistic_circuit=pc))
    #         return prod
    #     elif _attribute_kind(T, v) == 'relation_aggregate':
    #         prod.add_subcircuit(ExchangeableDistributionTemplate(Continuous(f"{C.__name__}.{v}"), probabilistic_circuit=pc)())
    #         return prod
    #     if _is_part(C, v):
    #         # Determine child class and gather child instances
    #         # Try to infer from available data in T
    #         childs: List[Any] = []
    #         for t in T:
    #             if not hasattr(t, v):
    #                 continue
    #             val = getattr(t, v)
    #             if isinstance(val, list):
    #                 childs.extend([x for x in val if x is not None])
    #             elif val is not None:
    #                 childs.append(val)
    #         if not childs:
    #             # no data → empty product
    #             return ProductUnit(probabilistic_circuit=pc)
    #         C_child = type(childs[0])
    #         return _learn_child(C_child, childs, cfg)
    #
    #     # Attribute or relation aggregate
    #     dist = _estimate_univariate(C, T, v, cfg)
    #     # Build a single-root product that owns the leaf and return it
    #     prod = ProductUnit(probabilistic_circuit=pc)
    #     prod.add_subcircuit(leaf(dist, probabilistic_circuit=pc))
    #     return prod
    #
    # # jpt = JPT()
    # # Try independence partition
    # parts = _independence_partition(C, T, V, cfg)
    # if parts is not None:
    #     prod = ProductUnit(probabilistic_circuit=pc)
    #     for Vj in parts:
    #         sub = LearnRSPN(C, T, Vj, cfg, probabilistic_circuit=pc)
    #         prod.add_subcircuit(sub)
    #     return prod
    #
    # # Otherwise, cluster instances and build sum
    # clusters = _cluster_instances(C, T, V, cfg)
    # if clusters is None:
    #     # Fallback: treat as independent singletons
    #     prod = ProductUnit(probabilistic_circuit=pc)
    #     for v in V:
    #         sub = LearnRSPN(C, T, [v], cfg, probabilistic_circuit=pc)
    #         prod.add_subcircuit(sub)
    #     return prod
    #
    # total = float(sum(len(ci) for ci in clusters))
    # summ = SumUnit(probabilistic_circuit=pc)
    # for Ti in clusters:
    #     weight = len(Ti) / total if total > 0 else 1.0 / len(clusters)
    #     sub = LearnRSPN(C, Ti, V, cfg, probabilistic_circuit=pc)
    #     summ.add_subcircuit(sub, log_weight=weight)
    # return summ

