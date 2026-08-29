from __future__ import annotations

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from semantic_digital_twin.spatial_types import Pose2D


def _fitted_pose_backend() -> ProbabilisticBackend:
    """
    A backend over a circuit fitted directly on bare ``Pose2D`` instances -- nothing
    wrapping it, so grounding has nothing to construct except the pose itself.
    """
    poses = [
        Pose2D(x=float(index), y=float(index) * 2, yaw=0.1 * index)
        for index in range(10)
    ]
    circuit = RelationalProbabilisticCircuit(Pose2D, min_samples_per_leaf=0.1).fit(
        [to_dao(pose) for pose in poses]
    )
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=circuit)
    return ProbabilisticBackend(model_registry=registry)


# %% minimal reproduction: grounding a free Pose2D through a real backend


def test_probabilistic_backend_evaluates_a_fully_free_pose2d() -> None:
    """
    Minimal reproduction of the grounding failure ``mode_query`` hits in
    ``shelf_placement.py``, stripped down to a bare ``Pose2D`` -- no shelf, no layer,
    no held object -- so the failure is attributable to ``Pose2D`` grounding itself
    and not to anything the shelf domain adds around it.

    ``_held_object_slot`` builds exactly this shape, ``a(Pose2D)(x=..., y=..., yaw=
    ...)``, to leave the held object's placement for the circuit's mode search to
    answer. Evaluating any query through :class:`ProbabilisticBackend` first calls
    ``RelationalProbabilisticCircuit.ground``, which calls ``query.construct_instance()``
    on the whole query before any field is resolved -- not because it needs the free
    field's value, but to build a concrete instance for its own aggregation-statistic
    bookkeeping. ``Pose2D.__init__`` converts its arguments into a casadi symbolic
    vector immediately, so it cannot construct with the ``Ellipsis`` placeholder an
    underspecified field carries, and grounding fails before the backend ever gets to
    answer the query.
    """
    backend = _fitted_pose_backend()
    query = a(Pose2D)(x=..., y=..., yaw=...)

    next(iter(query.evaluate(backend=backend)))
