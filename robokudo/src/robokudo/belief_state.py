"""
RoboKudo's belief state: the semantic digital twin world it perceives into.

The world is shared with other processes through a
:class:`~semantic_digital_twin.adapters.ros.world_synchronizer.WorldSynchronizer`.
Updates from other processes are applied only when a perception run starts, so the
world stays put while the annotators of a run work on it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
import numpy.typing as npt
from py_trees.blackboard import Blackboard
from typing_extensions import TYPE_CHECKING

from robokudo.defs import PACKAGE_NAME
from robokudo.identifier import BBIdentifier
from robokudo.types.belief_state import ObjectBeliefState
from semantic_digital_twin.adapters.ros.messages import (
    MetaData,
    ModificationBlock,
    WorldStateUpdate,
    WorldUpdate,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    RemoveKinematicStructureEntityModification,
    WorldModelModificationBlock,
)

if TYPE_CHECKING:
    from semantic_digital_twin.adapters.ros.world_synchronizer import (
        WorldSynchronizer,
    )

# %% snapshot of a run's start


@dataclass
class BeliefStateSnapshot:
    """
    What the belief state held when a perception run started.
    """

    modification_block_count: int
    """
    How many modification blocks the history of the world held.
    """

    positions: dict[UUID, float]
    """
    The position of every degree of freedom, by its id.
    """


# %% belief state


@dataclass
class BeliefState:
    """
    The world RoboKudo believes in, together with the objects it keeps track of.
    """

    world: World = field(default_factory=World)
    """
    The world perception results are written into.
    """

    synchronizer: WorldSynchronizer | None = None
    """
    Exchanges the changes of :attr:`world` with other processes; ``None`` when RoboKudo
    runs on its own.

    It is expected to defer incoming updates, so that :meth:`start_run` decides when
    they are applied.
    """

    tracked_objects: dict[UUID, ObjectBeliefState] = field(default_factory=dict)
    """
    The objects RoboKudo perceived, by the id of their body.
    """

    @classmethod
    def current(cls) -> BeliefState:
        """
        The belief state of this process.

        A belief state with an empty world is created when there is none yet.
        """
        blackboard = Blackboard()
        if not blackboard.exists(BBIdentifier.BELIEF_STATE):
            cls().make_current()
        return blackboard.get(BBIdentifier.BELIEF_STATE)

    def make_current(self) -> None:
        """
        Make this the belief state of this process, see :meth:`current`.
        """
        Blackboard().set(BBIdentifier.BELIEF_STATE, self)

    # %% perception runs

    def start_run(self) -> BeliefStateSnapshot:
        """
        Prepare the belief state for a new perception run.

        Applies the updates other processes sent since the last run and forgets the
        tracked objects whose body they removed.

        :return: What the belief state holds at the start of the run.
        """
        modification_block_count = len(self.world.modification_history)
        if self.synchronizer is not None:
            self.synchronizer.apply_missed_messages()
        for modification_block in self.world.modification_history[
            modification_block_count:
        ]:
            for modification in modification_block:
                if isinstance(modification, RemoveKinematicStructureEntityModification):
                    self.tracked_objects.pop(modification.kinematic_structure_id, None)

        return BeliefStateSnapshot(
            modification_block_count=len(self.world.modification_history),
            positions=self.world.state.to_uuid_position_dict(),
        )

    def changes_since(self, snapshot: BeliefStateSnapshot) -> WorldUpdate:
        """
        The changes made to the world since ``snapshot`` was taken.

        :param snapshot: What the belief state held at the start of a run.
        :return: The model changes as one modification block, and the degrees of freedom
            that are new or moved. When the world is synchronized, its stream position
            says up to which publication other processes have to catch up to hold these
            changes.
        """
        modifications = WorldModelModificationBlock(
            modifications=[
                modification
                for modification_block in self.world.modification_history[
                    snapshot.modification_block_count :
                ]
                for modification in modification_block
            ]
        )
        changed_positions = {
            degree_of_freedom_id: position
            for degree_of_freedom_id, position in self.world.state.to_uuid_position_dict().items()
            if degree_of_freedom_id not in snapshot.positions
            or not np.isclose(position, snapshot.positions[degree_of_freedom_id])
        }

        meta_data = self.publisher
        sequence_number = (
            0
            if self.synchronizer is None
            else self.synchronizer.published_sequence_number
        )
        return WorldUpdate(
            meta_data=meta_data,
            modification_block=ModificationBlock(
                meta_data=meta_data, modifications=modifications
            ),
            state_update=WorldStateUpdate(
                meta_data=meta_data,
                ids=list(changed_positions.keys()),
                states=list(changed_positions.values()),
            ),
            sequence_number=sequence_number,
        )

    @property
    def publisher(self) -> MetaData:
        """
        Who publishes the changes of this belief state.
        """
        if self.synchronizer is not None:
            return self.synchronizer.meta_data
        return MetaData(node_name=PACKAGE_NAME, process_id=os.getpid())

    # %% camera

    def set_camera_pose(
        self,
        world_frame: str,
        camera_frame: str,
        world_T_camera: HomogeneousTransformationMatrix | npt.NDArray[np.float64],
    ) -> HomogeneousTransformationMatrix:
        """
        Place the camera in the world.

        A frame the world does not hold yet is added. A camera frame that already
        belongs to the world, for example as a link of a robot, keeps its place in the
        kinematic structure, and its pose is left to whoever owns it.

        :param world_frame: The name of the frame the pose is expressed in.
        :param camera_frame: The name of the camera frame.
        :param world_T_camera: The pose of the camera in the world frame.
        :return: The pose of the camera, between the bodies of both frames.
        """
        self.add_camera_frame(world_frame=world_frame, camera_frame=camera_frame)
        world_body = self.world.get_body_by_name(world_frame)
        camera_body = self.world.get_body_by_name(camera_frame)
        bound_world_T_camera = HomogeneousTransformationMatrix(
            data=world_T_camera,
            reference_frame=world_body,
            child_frame=camera_body,
        )
        for connection in self.world.get_connections_by_name(
            self.camera_connection_name(world_frame, camera_frame)
        ):
            connection.origin = bound_world_T_camera
        return bound_world_T_camera

    def add_camera_frame(self, world_frame: str, camera_frame: str) -> None:
        """
        Add the bodies of the world and camera frames that the world does not hold yet.

        A camera body added here is connected to the world body, so its pose can be set
        with :meth:`set_camera_pose`.

        :param world_frame: The name of the world frame.
        :param camera_frame: The name of the camera frame.
        """
        has_world_body = len(self.world.get_bodies_by_name(world_frame)) > 0
        has_camera_body = len(self.world.get_bodies_by_name(camera_frame)) > 0
        if has_world_body and has_camera_body:
            return

        with self.world.modify_world():
            if not has_world_body:
                self.world.add_body(self.create_frame_body(world_frame))
            if has_camera_body:
                return
            camera_body = self.create_frame_body(camera_frame)
            self.world.add_body(camera_body)
            self.world.add_connection(
                Connection6DoF.create_with_dofs(
                    world=self.world,
                    parent=self.world.get_body_by_name(world_frame),
                    child=camera_body,
                    name=self.camera_connection_name(world_frame, camera_frame),
                )
            )

    @staticmethod
    def create_frame_body(
        frame: str, axis_length: float = 0.4, axis_thickness: float = 0.016
    ) -> Body:
        """
        Create a body that stands for a coordinate frame, drawn as its x, y and z axes
        in red, green and blue.

        :param frame: The name of the frame.
        :param axis_length: How long each axis is drawn, in meters.
        :param axis_thickness: How thick each axis is drawn, in meters.
        :return: The new body.
        """
        body = Body(name=PrefixedName(name=frame))
        half_axis_length = axis_length / 2
        body.visual.append(
            Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=half_axis_length, reference_frame=body
                ),
                scale=Scale(x=axis_length, y=axis_thickness, z=axis_thickness),
                color=Color.RED(),
            )
        )
        body.visual.append(
            Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=half_axis_length, reference_frame=body
                ),
                scale=Scale(x=axis_thickness, y=axis_length, z=axis_thickness),
                color=Color.GREEN(),
            )
        )
        body.visual.append(
            Box(
                origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=half_axis_length, reference_frame=body
                ),
                scale=Scale(x=axis_thickness, y=axis_thickness, z=axis_length),
                color=Color.BLUE(),
            )
        )
        return body

    @staticmethod
    def camera_connection_name(world_frame: str, camera_frame: str) -> PrefixedName:
        """
        The name of the connection that places a camera added by
        :meth:`add_camera_frame` in the world frame.

        :param world_frame: The name of the world frame.
        :param camera_frame: The name of the camera frame.
        """
        return PrefixedName(name=f"{camera_frame}_T_{world_frame}")
