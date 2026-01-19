from __future__ import annotations
import pandas as pd
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type
from probabilistic_model.probabilistic_circuit.relational.rspns import (
    RSPNTemplate,
)
from ...learning.jpt.jpt import JPT
from ...learning.jpt.variables import infer_variables_from_dataframe


def get_aggregate_statistics(instance: Any) -> List[Tuple[Any, str]]:
    statistics = []
    for name in dir(instance):
        if name.startswith("__"):
            continue

        attr = getattr(instance, name)

        if not callable(attr):
            continue

        if not hasattr(attr, "_statistic_name"):
            continue

        statistics.append((attr(), attr._statistic_name))

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

    for instance in instances:
        aggregation_values = get_aggregate_statistics(instance)
        assert isinstance(instance, cls)

        if len(class_spec["attributes"]) > 0:
            for attribute in class_spec["attributes"]:
                value = getattr(instance, attribute)
                if attribute not in df_data:
                    df_data[attribute] = [value]
                else:
                    df_data[attribute].append(value)

        for category in ["relations", "unique_parts", "exchangeable_parts"]:
            if len(class_spec[category]) > 0:
                for item in class_spec[category]:
                    values = [
                        value
                        for value, statistic_type in aggregation_values
                        if statistic_type == item
                    ]
                    if item not in df_data:
                        df_data[item] = values
                    else:
                        df_data[item] += values

    df = pd.DataFrame(df_data)
    variables = infer_variables_from_dataframe(df)
    jpt = JPT(variables)
    jpt = jpt.fit(df)
    rspn = RSPNTemplate(class_spec, jpt)
    # jpt.plot_structure()
    # plt.show()
    return rspn
