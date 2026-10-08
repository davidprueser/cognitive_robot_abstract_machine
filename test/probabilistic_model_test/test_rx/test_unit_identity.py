from __future__ import annotations

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
)


# %% identity comparison
def test_units_of_different_circuits_compare_unequal_without_raising() -> None:
    """
    Units compare by identity: comparing two units field by field would compare their
    circuits, which refuse equality checks.
    """
    first_unit = ProductUnit(probabilistic_circuit=ProbabilisticCircuit())
    second_unit = ProductUnit(probabilistic_circuit=ProbabilisticCircuit())

    assert first_unit != second_unit
    assert first_unit not in [second_unit]
