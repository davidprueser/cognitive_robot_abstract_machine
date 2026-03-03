from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, TYPE_CHECKING

import numpy as np
from line_profiler import profile

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.adapters.qp_adapter import GiskardToQPAdapter
from giskardpy.qp.qp_data import QPData, ZeroWeightQPDataFilter
from krrood.symbolic_math.symbolic_math import VariableParameters

if TYPE_CHECKING:
    pass


@dataclass
class GiskardToExplicitQPAdapter(GiskardToQPAdapter):
    """
    Takes free variables and constraints and converts them to a QP problem in the following format, depending on the
    class attributes:

    min_x 0.5 x^T H x + g^T x
    s.t.  lb <= x <= ub     (box constraints)
          Ex <= bE          (equality constraints)
          lbA <= Ax <= ubA  (lower/upper inequality constraints)
    """

    def general_qp_to_specific_qp(
        self,
        quadratic_weights: sm.Vector,
        linear_weights: sm.Vector,
        box_lower_constraints: sm.Vector,
        box_upper_constraints: sm.Vector,
        eq_matrix_dofs: sm.Matrix,
        eq_matrix_slack: sm.Matrix,
        eq_bounds: sm.Vector,
        neq_matrix_dofs: sm.Matrix,
        neq_matrix_slack: sm.Matrix,
        neq_lower_bounds: sm.Vector,
        neq_upper_bounds: sm.Vector,
    ):
        eq_matrix = sm.hstack(
            [
                eq_matrix_dofs,
                eq_matrix_slack,
                sm.Matrix.zeros(eq_matrix_slack.shape[0], self.num_neq_slack_variables),
            ]
        )
        neq_matrix = sm.hstack(
            [
                neq_matrix_dofs,
                sm.Matrix.zeros(neq_matrix_slack.shape[0], self.num_eq_slack_variables),
                neq_matrix_slack,
            ]
        )

        self.free_symbols = [
            self.world_state_symbols,
            self.life_cycle_symbols,
            self.float_variables,
        ]

        self.eq_matrix_compiled = eq_matrix.compile(
            parameters=VariableParameters.from_lists(*self.free_symbols),
            sparse=self.sparse,
        )
        self.neq_matrix_compiled = neq_matrix.compile(
            parameters=VariableParameters.from_lists(*self.free_symbols),
            sparse=self.sparse,
        )

        self.combined_vector_f = sm.CompiledFunctionWithViews(
            expressions=[
                quadratic_weights,
                linear_weights,
                box_lower_constraints,
                box_upper_constraints,
                eq_bounds,
                neq_lower_bounds,
                neq_upper_bounds,
            ],
            parameters=VariableParameters.from_lists(*self.free_symbols),
        )

    def evaluate(
        self,
        world_state: np.ndarray,
        life_cycle_state: np.ndarray,
        float_variables: np.ndarray,
    ) -> QPData:
        args = [
            world_state,
            life_cycle_state,
            float_variables,
        ]
        eq_matrix_np_raw = self.eq_matrix_compiled(*args)
        neq_matrix_np_raw = self.neq_matrix_compiled(*args)
        (
            quadratic_weights_np_raw,
            linear_weights_np_raw,
            box_lower_constraints_np_raw,
            box_upper_constraints_np_raw,
            eq_bounds_np_raw,
            neq_lower_bounds_np_raw,
            neq_upper_bounds_np_raw,
        ) = self.combined_vector_f(*args)

        return QPData(
            quadratic_weights=quadratic_weights_np_raw,
            linear_weights=linear_weights_np_raw,
            box_lower_constraints=box_lower_constraints_np_raw,
            box_upper_constraints=box_upper_constraints_np_raw,
            eq_matrix=eq_matrix_np_raw,
            eq_bounds=eq_bounds_np_raw,
            neq_matrix=neq_matrix_np_raw,
            neq_lower_bounds=neq_lower_bounds_np_raw,
            neq_upper_bounds=neq_upper_bounds_np_raw,
        )
