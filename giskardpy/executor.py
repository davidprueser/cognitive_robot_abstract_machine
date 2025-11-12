from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Optional

from giskardpy.data_types.exceptions import EmptyProblemException
from giskardpy.model.collision_world_syncer import CollisionWorldSynchronizer
from giskardpy.motion_statechart.auxilary_variable_manager import (
    AuxiliaryVariableManager,
)
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.qp.qp_controller import QPController
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World


@dataclass
class Executor:
    motion_statechart: MotionStatechart
    world: World
    controller_config: Optional[QPControllerConfig] = None
    collision_scene: Optional[CollisionWorldSynchronizer] = None
    auxiliary_variable_manager: AuxiliaryVariableManager = field(
        default_factory=AuxiliaryVariableManager, init=False
    )
    qp_controller: Optional[QPController] = field(default=None, init=False)

    def compile(self):
        context = BuildContext(
            world=self.world,
            auxiliary_variable_manager=self.auxiliary_variable_manager,
            collision_scene=self.collision_scene,
        )
        self.motion_statechart.compile(context)
        self._compile_qp_controller(self.controller_config)

    @property
    def build_context(self) -> BuildContext:
        return BuildContext(
            world=self.world,
            auxiliary_variable_manager=self.auxiliary_variable_manager,
            collision_scene=self.collision_scene,
        )

    def tick(self):
        self.motion_statechart.tick(self.build_context)
        if self.qp_controller is None:
            return
        next_cmd = self.qp_controller.get_cmd(
            world_state=self.world.state.data,
            life_cycle_state=self.motion_statechart.life_cycle_state.data,
            external_collisions=np.array([], dtype=np.float64),
            self_collisions=np.array([], dtype=np.float64),
            auxiliary_variables=self.auxiliary_variable_manager.resolve_auxiliary_variables(),
        )
        self.world.apply_control_commands(
            next_cmd,
            self.qp_controller.config.control_dt or self.qp_controller.config.mpc_dt,
            self.qp_controller.config.max_derivative,
        )

    def tick_until_end(self, timeout: int = 1_000):
        """
        Calls tick until is_end_motion() returns True.
        :param timeout: Max number of ticks to perform.
        """
        for i in range(timeout):
            self.tick()
            if self.motion_statechart.is_end_motion():
                return
        raise TimeoutError("Timeout reached while waiting for end of motion.")

    def _compile_qp_controller(self, controller_config: QPControllerConfig):
        ordered_dofs = sorted(
            self.world.active_degrees_of_freedom,
            key=lambda dof: self.world.state._index[dof.name],
        )
        constraint_collection = (
            self.motion_statechart.combine_constraint_collections_of_nodes()
        )
        if len(constraint_collection.constraints) == 0:
            self.qp_controller = None
            # to not build controller, if there are no constraints
            return
        self.qp_controller = QPController(
            config=controller_config,
            degrees_of_freedom=ordered_dofs,
            constraint_collection=constraint_collection,
            world_state_symbols=self.world.state.get_variables(),
            life_cycle_variables=self.motion_statechart.life_cycle_state.life_cycle_symbols(),
            auxiliary_variables=self.auxiliary_variable_manager.variables,
        )
        if self.qp_controller.has_not_free_variables():
            raise EmptyProblemException(
                "Tried to compile a QPController without free variables."
            )
