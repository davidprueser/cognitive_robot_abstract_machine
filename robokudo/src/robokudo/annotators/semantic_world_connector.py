from __future__ import annotations

from timeit import default_timer

import numpy as np
from py_trees.common import Status
from scipy.optimize import linear_sum_assignment

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.belief_state import BeliefState
from robokudo.cas import CAS, CASViews
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    PoseAnnotation,
    StampedPoseAnnotation,
)
from robokudo.types.belief_state import ObjectBeliefState
from robokudo.types.cv import ImageROI
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import (
    draw_bounding_boxes_from_object_hypotheses,
    transform_pose_from_camera_to_world,
)
from robokudo.utils.hypothesis_comparators import ObjectHypothesisComparator
from robokudo.utils.transform import (
    get_quaternion_from_transform_matrix,
    get_transform_matrix_from_q,
    get_translation_from_transform_matrix,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.world_entity import Body


class SemanticDigitalTwinConnector(ThreadedAnnotator):
    """
    Write the perceived objects into the belief state.

    Object hypotheses are associated with the objects the belief state tracks. Objects
    seen for the first time are added to the world, and every perceived object is moved
    to where it was seen.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self) -> None:
                """
                Initialize a new set of parameters for the descriptor.
                """
                self.confidence_threshold = 0.05

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "SemanticDigitalTwinSynchronization",
        descriptor: SemanticDigitalTwinConnector.Descriptor | None = None,
    ) -> None:
        """
        Default construction.

        Minimal one-time init!
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.object_comparator = (
            ObjectHypothesisComparator(self.get_cas)
            .with_comparator_for(ImageROI, weight=0.2)
            .with_comparator_for(StampedPoseAnnotation, weight=0.4)
            .with_comparator_for(BoundingBox3DAnnotation, weight=0.4)
        )

    def compute(self) -> Status:
        """
        Write the objects perceived in this run into the belief state.
        """
        start_timer = default_timer()

        cas = self.get_cas()
        object_hypotheses: list[ObjectHypothesis] = cas.filter_annotations_by_type(
            ObjectHypothesis
        )
        object_beliefs = list(BeliefState.current().tracked_objects.values())

        associated_hypotheses = self.associate_hypotheses_with_beliefs(
            object_hypotheses, object_beliefs, cas
        )
        self.place_in_world(
            [object_belief for _, object_belief in associated_hypotheses], cas
        )
        self.create_association_visualization(associated_hypotheses)
        self.log_world_state()

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    def associate_hypotheses_with_beliefs(
        self,
        object_hypotheses: list[ObjectHypothesis],
        object_beliefs: list[ObjectBeliefState],
        cas: CAS,
    ) -> list[tuple[ObjectHypothesis, ObjectBeliefState]]:
        """
        Associate current object hypotheses with existing or new object beliefs.

        New object beliefs are tracked by the belief state, but not yet placed in its
        world, see :meth:`place_in_world`.
        """
        if len(object_hypotheses) == 0:
            return []

        # Association uses the newest pose annotation from each hypothesis.
        self.add_world_pose_annotations(object_hypotheses, cas)

        if len(object_beliefs) == 0:
            return self.create_new_beliefs_for_hypotheses(object_hypotheses)

        cost_matrix = self.create_association_cost_matrix(
            object_hypotheses, object_beliefs
        )
        hypothesis_indices, belief_indices = linear_sum_assignment(cost_matrix)

        associated_hypotheses = self.apply_assignment(
            object_hypotheses,
            object_beliefs,
            cost_matrix,
            hypothesis_indices,
            belief_indices,
        )
        associated_hypotheses.extend(
            self.create_new_beliefs_for_unmatched_hypotheses(
                object_hypotheses, set(hypothesis_indices)
            )
        )
        return associated_hypotheses

    def add_world_pose_annotations(
        self, object_hypotheses: list[ObjectHypothesis], cas: CAS
    ) -> None:
        """
        Add world-frame stamped poses to hypotheses when a camera pose is available.
        """
        if cas.camera_to_world_transform is None:
            return

        for object_hypothesis in object_hypotheses:
            pose_annotation = self.get_latest_pose_annotation(object_hypothesis)
            if pose_annotation is None:
                continue

            pose_in_world = transform_pose_from_camera_to_world(cas, pose_annotation)
            stamped_pose = StampedPoseAnnotation()
            stamped_pose.source = self.get_class_name()
            stamped_pose.translation = pose_in_world.translation
            stamped_pose.rotation = pose_in_world.rotation
            if cas.world_frame is not None:
                stamped_pose.frame = cas.world_frame
            if cas.contains(CASViews.CAMERA_INFO):
                stamped_pose.timestamp = cas.get(CASViews.CAMERA_INFO).header.stamp

            object_hypothesis.annotations.append(stamped_pose)

    @staticmethod
    def get_latest_pose_annotation(
        object_hypothesis: ObjectHypothesis,
    ) -> PoseAnnotation | None:
        """
        Return the latest plain pose annotation for a hypothesis.
        """
        for annotation in reversed(object_hypothesis.annotations):
            if type(annotation) is PoseAnnotation:
                return annotation
        return None

    def create_association_cost_matrix(
        self,
        object_hypotheses: list[ObjectHypothesis],
        object_beliefs: list[ObjectBeliefState],
    ) -> np.ndarray:
        """
        Create a negative-similarity cost matrix for Hungarian assignment.
        """
        cost_matrix = np.full((len(object_hypotheses), len(object_beliefs)), 1e9)
        for hypothesis_idx, object_hypothesis in enumerate(object_hypotheses):
            for belief_idx, object_belief in enumerate(object_beliefs):
                similarity = self.object_comparator.compute_similarity(
                    object_hypothesis, object_belief.latest_hypothesis
                )
                cost_matrix[hypothesis_idx][belief_idx] = -similarity
        return cost_matrix

    def apply_assignment(
        self,
        object_hypotheses: list[ObjectHypothesis],
        object_beliefs: list[ObjectBeliefState],
        cost_matrix: np.ndarray,
        hypothesis_indices: np.ndarray,
        belief_indices: np.ndarray,
    ) -> list[tuple[ObjectHypothesis, ObjectBeliefState]]:
        """
        Update matched beliefs or create new beliefs for low-confidence matches.
        """
        associated_hypotheses: list[tuple[ObjectHypothesis, ObjectBeliefState]] = []
        threshold = self.descriptor.parameters.confidence_threshold

        for hypothesis_idx, belief_idx in zip(hypothesis_indices, belief_indices):
            object_hypothesis = object_hypotheses[hypothesis_idx]
            object_belief = object_beliefs[belief_idx]
            similarity = -cost_matrix[hypothesis_idx, belief_idx]

            if similarity > threshold:
                object_belief.add_hypothesis(object_hypothesis)
            else:
                object_belief = self.track_new_object(object_hypothesis)
            associated_hypotheses.append((object_hypothesis, object_belief))

        return associated_hypotheses

    def create_new_beliefs_for_hypotheses(
        self, object_hypotheses: list[ObjectHypothesis]
    ) -> list[tuple[ObjectHypothesis, ObjectBeliefState]]:
        """
        Create object beliefs for all given hypotheses.
        """
        return [
            (object_hypothesis, self.track_new_object(object_hypothesis))
            for object_hypothesis in object_hypotheses
        ]

    def create_new_beliefs_for_unmatched_hypotheses(
        self,
        object_hypotheses: list[ObjectHypothesis],
        matched_hypothesis_indices: set[int],
    ) -> list[tuple[ObjectHypothesis, ObjectBeliefState]]:
        """
        Create object beliefs for hypotheses not returned by Hungarian assignment.
        """
        unmatched_hypotheses = [
            object_hypothesis
            for hypothesis_idx, object_hypothesis in enumerate(object_hypotheses)
            if hypothesis_idx not in matched_hypothesis_indices
        ]
        # Hungarian assignment only pairs min(len(hypotheses), len(beliefs)); extras start new beliefs.
        return self.create_new_beliefs_for_hypotheses(unmatched_hypotheses)

    @staticmethod
    def track_new_object(object_hypothesis: ObjectHypothesis) -> ObjectBeliefState:
        """
        Start tracking an object that was seen for the first time.

        :param object_hypothesis: How the object was seen.
        :return: The belief about the new object.
        """
        object_belief = ObjectBeliefState.create_with_new_body().add_hypothesis(
            object_hypothesis
        )
        BeliefState.current().tracked_objects[object_belief.uuid] = object_belief
        return object_belief

    def place_in_world(self, object_beliefs: list[ObjectBeliefState], cas: CAS) -> None:
        """
        Place the objects of the given beliefs in the world of the belief state, at the
        pose of their latest 3D bounding box.

        Objects the world does not hold yet are added as a box of that size, all in one
        modification. The poses of all objects are then set in one state change. Objects
        without a 3D bounding box, or seen without a camera pose in the world, are left
        out.

        :param object_beliefs: The beliefs whose objects were perceived.
        :param cas: The CAS that holds the camera pose of the perception run.
        """
        if cas.world_frame is None or cas.camera_to_world_transform is None:
            return
        world = BeliefState.current().world
        world_body = world.get_body_by_name(cas.world_frame)
        placed_beliefs = [
            object_belief
            for object_belief in object_beliefs
            if object_belief.latest_bbox_3d is not None
        ]
        new_beliefs = [
            object_belief
            for object_belief in placed_beliefs
            if world.find_world_entity_with_id(object_belief.uuid) is None
        ]

        if new_beliefs:
            with world.modify_world():
                for object_belief in new_beliefs:
                    self.add_to_world(object_belief, world, world_body)

        with world.batch_state_changes():
            for object_belief in placed_beliefs:
                object_belief.body.parent_connection.origin = self.world_origin(
                    object_belief, cas, world_body
                )

    @staticmethod
    def add_to_world(
        object_belief: ObjectBeliefState, world: World, world_body: Body
    ) -> None:
        """
        Add the body of an object belief to the world as a box of the size of its latest
        3D bounding box, free to move relative to the world body.

        Must be called inside a modification of ``world``.

        :param object_belief: The belief whose body is added.
        :param world: The world the body is added to.
        :param world_body: The body of the world frame.
        """
        body = object_belief.body
        bounding_box = object_belief.latest_bbox_3d
        body.visual.append(
            Box(
                origin=HomogeneousTransformationMatrix(reference_frame=body),
                scale=Scale(
                    x=bounding_box.x_length,
                    y=bounding_box.y_length,
                    z=bounding_box.z_length,
                ),
            )
        )
        world.add_body(body)
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=world_body, child=body)
        )

    @staticmethod
    def world_origin(
        object_belief: ObjectBeliefState, cas: CAS, world_body: Body
    ) -> HomogeneousTransformationMatrix:
        """
        The pose of the latest 3D bounding box of an object belief in the world frame.

        :param object_belief: The belief whose pose is computed.
        :param cas: The CAS that holds the camera pose the bounding box was seen from.
        :param world_body: The body of the world frame.
        :return: The pose of the object's body relative to the world body.
        """
        bounding_box_pose = object_belief.latest_bbox_3d.pose
        world_T_object = np.matmul(
            cas.camera_to_world_transform.to_np(),
            get_transform_matrix_from_q(
                bounding_box_pose.rotation, bounding_box_pose.translation
            ),
        )
        translation = get_translation_from_transform_matrix(world_T_object)
        rotation = get_quaternion_from_transform_matrix(world_T_object)
        return HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=translation[0],
            pos_y=translation[1],
            pos_z=translation[2],
            quat_x=rotation[0],
            quat_y=rotation[1],
            quat_z=rotation[2],
            quat_w=rotation[3],
            reference_frame=world_body,
            child_frame=object_belief.body,
        )

    def log_world_state(self) -> None:
        """
        Log a compact summary of the current SemDT world contents.
        """
        belief_world = BeliefState.current().world
        self.rk_logger.debug(
            f"SemDT \nKS Entities: {len(belief_world.kinematic_structure_entities)}\nViews: {len(belief_world.semantic_annotations)}\nConnections: {len(belief_world.connections)}"
        )

    def create_association_visualization(
        self,
        associated_hypotheses: list[tuple[ObjectHypothesis, ObjectBeliefState]],
    ) -> None:
        """
        Publish a 2D image showing which SemDT UUID each hypothesis maps to.
        """
        cas = self.get_cas()
        if not cas.contains(CASViews.COLOR_IMAGE):
            return

        visualization_img = cas.get_copy(CASViews.COLOR_IMAGE)
        object_hypotheses = [oh for oh, _ in associated_hypotheses]
        labels_by_hypothesis_id = {
            id(oh): self.get_object_belief_label(object_belief)
            for oh, object_belief in associated_hypotheses
        }

        draw_bounding_boxes_from_object_hypotheses(
            visualization_img,
            object_hypotheses,
            lambda oh: labels_by_hypothesis_id.get(id(oh), "unassociated"),
        )
        self.get_annotator_output_struct().set_image(visualization_img)

    @staticmethod
    def get_object_belief_label(object_belief: ObjectBeliefState) -> str:
        """
        Return a compact display label for an object belief UUID.
        """
        return f"{str(object_belief.uuid)[:10]}..."
