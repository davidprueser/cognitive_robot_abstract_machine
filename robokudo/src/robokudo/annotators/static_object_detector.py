"""
Robokudo Static Object Detector Module

This module provides functionality for detecting objects at predefined locations
using either manually configured bounding boxes or a world descriptor.
The detector can create object hypotheses with poses, masks, and class labels.

.. note::
   All poses are defined relative to the camera frame by default unless pose_in_world_coordinates is True.
"""

from __future__ import annotations

import copy
from enum import Enum
from timeit import default_timer

import cv2
import numpy as np
from py_trees.common import Status
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import TYPE_CHECKING, Dict, List, Optional

import robokudo.world as robokudo_world
from robokudo.annotators.core import BaseAnnotator
from robokudo.common_analysis_structure import CommonAnalysisStructureViews
from robokudo.exceptions import ColorToDepthRatioMissing, UnknownMode
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    Classification,
    PoseAnnotation,
)
from robokudo.types.cv import ImageRegionOfInterest, Point2D, Rectangle
from robokudo.types.scene import ObjectHypothesis
from robokudo.types.tf import Pose
from robokudo.utils.annotator_helper import (
    get_world_to_camera_transform_matrix,
    scale_camera_intrinsics,
)
from robokudo.utils.cv_helper import (
    clamp_bounding_rectangle,
    get_scaled_color_image_for_depth_image,
    rectangle_outside_image,
)
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from robokudo.utils.o3d_helper import (
    get_2d_bounding_rectangle_from_3d_bounding_box,
    get_cloud_from_color_depth_and_mask,
    get_oriented_bounding_box_from_size_and_transform,
)
from robokudo.utils.transform import (
    get_quaternion_from_rotation_matrix,
    get_quaternion_from_transform_matrix,
    get_rotation_matrix_from_euler_angles,
    get_transform_matrix_from_translation,
    get_translation_from_transform_matrix,
)
from robokudo.world_descriptor import PredefinedObject

if TYPE_CHECKING:
    import numpy.typing as npt
    import open3d as o3d


class StaticObjectMode(Enum):
    BOUNDING_BOX = "bounding_box"
    WORLD_DESCRIPTOR = "world_descriptor"


class StaticObjectDetectorAnnotator(BaseAnnotator):
    """Find a cluster based on a preconfigured Bounding Box, Pose and Class name.

    This annotator can:

    * Create object hypotheses at fixed locations using manual bounding box coordinates
    * Read predefined objects from the shared world to automatically infer bounding boxes
    * Generate pose annotations in either camera or world coordinates
    * Create masks for the detected regions

    .. note::
       The detector supports both Euler angles and quaternions for rotation specification.
       Parameters can be dynamically reconfigured through ROS dynamic reconfigure.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for the StaticObjectDetectorAnnotator.

        Defines all configurable parameters including:

        * Bounding box dimensions and position
        * Object knowledge database settings
        * Pose generation options
        * Coordinate frame settings
        """

        class Parameters:
            """Parameters controlling the static object detection behavior."""

            def __init__(self) -> None:
                self.bounding_box_x: int = 1
                self.bounding_box_y: int = 1
                self.bounding_box_width: int = 10
                self.bounding_box_height: int = 10

                self.mode: StaticObjectMode = StaticObjectMode.BOUNDING_BOX
                """Defines the mode which mainly decide which sources of information are used to generate the Object Hypothesis"""

                # # If setting this to True, we detect a certain object stored in the ObjectKnowledgeBase
                # # This allows us to automatically infer the BoundingBox coordinates
                # self.detect_object_from_object_knowledge = False

                self.class_name: str = "unknown"
                """
                Define the class_name which is used for the object of interest
                Only used for StaticObjectMode.BOUNDING_BOX
                """

                self.class_names: List[str] = []
                """Used for StaticObjectMode.WORLD_DESCRIPTOR"""

                self.create_pose_annotation: bool = False
                """If True a Pose will be created for the object. Pose is relative to the camera frame by default.

                Only effective in Mode=StaticObjectDetectorAnnotator.Mode.WORLD_DESCRIPTOR
                """

                self.create_bounding_box_annotation: bool = False
                """If True a BoundingBox will be created for the object.

                Only effective in Mode=StaticObjectDetectorAnnotator.Mode.WORLD_DESCRIPTOR
                """

                self.create_mask: bool = True
                """If this is a true, a mask based on the region of interest will be generated that marks every pixel as ON"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "StaticObjectDetector",
        descriptor: StaticObjectDetectorAnnotator.Descriptor | None = None,
    ) -> None:
        """Default construction. Minimal one-time init!

        :param name: Name of the annotator instance
        :param descriptor: Configuration descriptor
        """
        super().__init__(name, descriptor)
        self.robokudo_logger.debug("%s.__init__()" % self.__class__.__name__)
        self.color: Optional[npt.NDArray] = None
        self.depth: Optional[npt.NDArray] = None
        self.cloud: Optional[o3d.geometry.PointCloud] = None
        self.camera_intrinsics = None
        self.object_body: Optional[Body] = None
        self.object_bodies_by_name: Dict[str, Body] = {}

    def detect_from_bounding_box_descriptor(
        self, color_in_red_green_blue: npt.NDArray
    ) -> ObjectHypothesis:
        """
        Detect only based on the bounding box.

        :param color_in_red_green_blue: Image in RGB order
        """
        object_hypothesis = ObjectHypothesis()
        object_hypothesis.id = str(0)
        object_hypothesis.source = self.name

        object_hypothesis.points = (
            self.get_cloud_from_2d_bounding_box_region_of_interest(
                color_in_red_green_blue
            )
        )
        # Use the non-scaled bounding box coordinates relative to the color image
        object_hypothesis.region_of_interest.rectangle.position.x = (
            self.descriptor.parameters.bounding_box_x
        )
        object_hypothesis.region_of_interest.rectangle.position.y = (
            self.descriptor.parameters.bounding_box_y
        )
        object_hypothesis.region_of_interest.rectangle.width = (
            self.descriptor.parameters.bounding_box_width
        )
        object_hypothesis.region_of_interest.rectangle.height = (
            self.descriptor.parameters.bounding_box_height
        )
        if self.descriptor.parameters.create_mask:
            object_hypothesis.region_of_interest.mask = (
                np.ones(
                    (
                        object_hypothesis.region_of_interest.rectangle.height,
                        object_hypothesis.region_of_interest.rectangle.width,
                    ),
                    dtype=np.uint8,
                )
                * 255
            )
        StaticObjectDetectorAnnotator.add_classification_annotation(
            object_hypothesis=object_hypothesis,
            class_name=self.descriptor.parameters.class_name,
        )
        return object_hypothesis

    @staticmethod
    def _body_name(body: Body) -> str:
        return body.name.name

    @staticmethod
    def _select_body_shape_collection(body: Body) -> ShapeCollection | None:
        if body.collision is not None and len(body.collision) > 0:
            return body.collision
        if body.visual is not None and len(body.visual) > 0:
            return body.visual
        return None

    def _get_body_bounding_box_size_and_center(
        self, body: Body
    ) -> tuple[np.ndarray, np.ndarray] | None:
        shape_collection = self._select_body_shape_collection(body)
        if shape_collection is None or len(shape_collection) == 0:
            self.robokudo_logger.warning(
                "Body %s has no collision or visual shapes; skipping.",
                self._body_name(body),
            )
            return None

        bounding_box = shape_collection.as_bounding_box_collection_in_frame(
            body
        ).bounding_box()
        size = np.array(
            [
                bounding_box.max_x - bounding_box.min_x,
                bounding_box.max_y - bounding_box.min_y,
                bounding_box.max_z - bounding_box.min_z,
            ],
            dtype=float,
        )
        center = np.array(
            [
                (bounding_box.min_x + bounding_box.max_x) / 2.0,
                (bounding_box.min_y + bounding_box.max_y) / 2.0,
                (bounding_box.min_z + bounding_box.max_z) / 2.0,
            ],
            dtype=float,
        )
        return size, center

    def detect_from_body(
        self,
        body: Body,
        object_id: int = 0,
        world_to_camera_transform_matrix: np.ndarray | None = None,
    ) -> ObjectHypothesis | None:
        """Detect from singular, passed semantic digital twin Body instance"""
        object_hypothesis = ObjectHypothesis()
        object_hypothesis.id = str(object_id)
        object_hypothesis.source = self.name
        object_hypothesis.object_knowledge = body

        if world_to_camera_transform_matrix is None:
            world_to_camera_transform_matrix = get_world_to_camera_transform_matrix(
                self.get_common_analysis_structure()
            )

        body_bounding_box = self._get_body_bounding_box_size_and_center(body)
        if body_bounding_box is None:
            return None
        bounding_box_size, bounding_box_center_body = body_bounding_box

        world_T_body = body.global_pose.to_np()
        body_T_bounding_box_center = get_transform_matrix_from_translation(
            bounding_box_center_body
        )
        world_T_bounding_box = world_T_body @ body_T_bounding_box_center
        bounding_box_transform_in_camera = (
            world_to_camera_transform_matrix @ world_T_bounding_box
        )

        # Calculate Bounding Box and resulting 2D Image Corner points based on pose in camera coordinates

        oriented_bounding_box = get_oriented_bounding_box_from_size_and_transform(
            bounding_box_size, bounding_box_transform_in_camera
        )
        corner_points = get_2d_bounding_rectangle_from_3d_bounding_box(
            self.get_common_analysis_structure(), oriented_bounding_box
        )

        image_height = (
            self.get_common_analysis_structure()
            .get(CommonAnalysisStructureViews.COLOR_IMAGE)
            .shape[0]
        )
        image_width = (
            self.get_common_analysis_structure()
            .get(CommonAnalysisStructureViews.COLOR_IMAGE)
            .shape[1]
        )

        if rectangle_outside_image(corner_points, image_width, image_height):
            self.robokudo_logger.info(
                f"region of interest of object {self._body_name(body)} would be completely out of camera frame. Skipping ..."
            )
            return None

        corner_points = clamp_bounding_rectangle(
            corner_points, image_width=image_width, image_height=image_height
        )

        region_of_interest = ImageRegionOfInterest()
        region_of_interest.rectangle.position.x = corner_points[0]
        region_of_interest.rectangle.position.y = corner_points[1]
        region_of_interest.rectangle.width = corner_points[2]
        region_of_interest.rectangle.height = corner_points[3]
        object_hypothesis.region_of_interest = region_of_interest
        object_hypothesis.points = self.cloud.crop(oriented_bounding_box)

        object_translation_in_camera = list(
            get_translation_from_transform_matrix(bounding_box_transform_in_camera)
        )
        object_rotation_in_camera = list(
            get_quaternion_from_transform_matrix(bounding_box_transform_in_camera)
        )

        if self.descriptor.parameters.create_pose_annotation:
            pose_annotation = PoseAnnotation()
            pose_annotation.source = "StaticObjectDetectorAnnotator"
            pose_annotation.translation = object_translation_in_camera
            pose_annotation.rotation = object_rotation_in_camera

            object_hypothesis.annotations.append(pose_annotation)

        if self.descriptor.parameters.create_bounding_box_annotation:
            bounding_box_annotation = BoundingBox3DAnnotation()
            bounding_box_annotation.source = "StaticObjectDetectorAnnotator"
            bounding_box_annotation.pose = Pose()
            bounding_box_annotation.pose.translation = object_translation_in_camera
            bounding_box_annotation.pose.rotation = object_rotation_in_camera

            bounding_box_annotation.x_length = float(bounding_box_size[0])
            bounding_box_annotation.y_length = float(bounding_box_size[1])
            bounding_box_annotation.z_length = float(bounding_box_size[2])

            object_hypothesis.annotations.append(bounding_box_annotation)

        StaticObjectDetectorAnnotator.add_classification_annotation(
            object_hypothesis=object_hypothesis, class_name=self._body_name(body)
        )

        return object_hypothesis

    def detect_from_body_base(
        self, world_to_camera_transform_matrix: Optional[npt.NDArray] = None
    ) -> List[ObjectHypothesis]:
        """
        Detect from a completed world descriptor

        :return: A list of ObjectHypothesis objects
        """
        object_hypotheses = []
        for class_name in self.descriptor.parameters.class_names:
            body = self.object_bodies_by_name.get(class_name)
            if body is None:
                continue
            object_hypothesis = self.detect_from_body(
                body, world_to_camera_transform_matrix=world_to_camera_transform_matrix
            )
            if object_hypothesis is not None:
                object_hypotheses.append(object_hypothesis)

        return object_hypotheses

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """Process current scene to detect configured static objects.

        Steps:

        * Gets current color image, depth image and point cloud
        * If using a world descriptor, validates object class exists
        * Scales color image to match depth image
        * Creates object hypothesis with:

          * Bounding box from configuration or world descriptor
          * Pose annotation if enabled
          * Point cloud from region of interest
          * Mask if enabled

        :return: SUCCESS if detection completed, FAILURE if required transforms not found
        :raises ColorToDepthRatioMissing: If color-to-depth ratio is not set
        :raises UnknownMode: If the configured static object mode is unsupported
        """
        start_timer = default_timer()

        self.color = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.COLOR_IMAGE
        )
        self.depth = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.DEPTH_IMAGE
        )
        self.cloud = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.CLOUD
        )
        self.camera_intrinsics = copy.deepcopy(
            self.get_common_analysis_structure().get(
                CommonAnalysisStructureViews.CAMERA_INTRINSIC
            )
        )

        world_frame_required = False
        world_to_camera_transform_matrix = None
        if self.descriptor.parameters.mode == StaticObjectMode.WORLD_DESCRIPTOR:
            predefined_object_annotations = (
                robokudo_world.world_instance().get_semantic_annotations_by_type(
                    PredefinedObject
                )
            )
            object_bodies = [
                annotation.body
                for annotation in predefined_object_annotations
                if annotation.body is not None
            ]
            self.object_bodies_by_name = {
                self._body_name(body): body for body in object_bodies
            }
            self.robokudo_logger.info(
                f"Loaded shared world descriptor bodies: {list(self.object_bodies_by_name.keys())}"
            )

            # Do some sanity checks and quit early if necessary
            for class_name in self.descriptor.parameters.class_names:
                if class_name not in self.object_bodies_by_name:
                    self.feedback_message = (
                        f"Couldn't find {class_name} in world descriptor"
                    )
                    self.robokudo_logger.warning(self.feedback_message)
                    return Status.SUCCESS

            world_frame_required = True

        if world_frame_required:
            try:
                world_to_camera_transform_matrix = get_world_to_camera_transform_matrix(
                    self.get_common_analysis_structure()
                )
            except:
                self.robokudo_logger.warning(
                    "Couldn't find world-to-camera transform in the common analysis structure"
                )
                return Status.FAILURE

        # Scale the image down so that it matches the depth image size
        try:
            resized_color = get_scaled_color_image_for_depth_image(
                self.get_common_analysis_structure(), self.color
            )
            scale_camera_intrinsics(self)
        except ColorToDepthRatioMissing:
            self.robokudo_logger.error(
                "No color to depth ratio set by your camera driver! Can't scale image for Point Cloud creation."
            )
            raise

        color_in_red_green_blue = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)

        object_hypotheses = []

        if self.descriptor.parameters.mode == StaticObjectMode.BOUNDING_BOX:
            object_hypothesis = self.detect_from_bounding_box_descriptor(
                color_in_red_green_blue
            )
            if object_hypothesis is None:
                return Status.SUCCESS
            object_hypotheses.append(object_hypothesis)
        elif self.descriptor.parameters.mode == StaticObjectMode.WORLD_DESCRIPTOR:
            object_hypotheses = self.detect_from_body_base(
                world_to_camera_transform_matrix=world_to_camera_transform_matrix
            )
            if len(object_hypotheses) == 0:
                # Simply return early but don't die
                return Status.SUCCESS
        else:
            raise UnknownMode(
                mode=self.descriptor.parameters.mode,
                context="StaticObjectDetectorAnnotator",
            )

        self.get_common_analysis_structure().annotations.extend(object_hypotheses)

        #
        # Create visualization Output
        #
        visualization_image = copy.deepcopy(self.color)

        for object_hypothesis in object_hypotheses:
            assert isinstance(object_hypothesis, ObjectHypothesis)
            object_hypothesis_region_of_interest = (
                object_hypothesis.region_of_interest.rectangle
            )
            upper_left = (
                object_hypothesis_region_of_interest.position.x,
                object_hypothesis_region_of_interest.position.y,
            )
            upper_left_text = (
                object_hypothesis_region_of_interest.position.x,
                object_hypothesis_region_of_interest.position.y - 5,
            )

            font = cv2.FONT_HERSHEY_COMPLEX
            visualization_image = cv2.putText(
                visualization_image,
                f"ROI-{object_hypothesis.id}({len(object_hypothesis.points.points)})",
                upper_left_text,
                font,
                0.5,
                (0, 0, 255),
                1,
                2,
            )
            visualization_image = cv2.rectangle(
                visualization_image,
                upper_left,
                (
                    object_hypothesis_region_of_interest.position.x
                    + object_hypothesis_region_of_interest.width,
                    object_hypothesis_region_of_interest.position.y
                    + object_hypothesis_region_of_interest.height,
                ),
                (0, 0, 255),
                2,
            )

            self.get_annotator_output_structure().set_geometries(
                object_hypothesis.points
            )

        self.get_annotator_output_structure().set_image(visualization_image)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS

    @staticmethod
    def add_classification_annotation(
        object_hypothesis: ObjectHypothesis, class_name: str
    ) -> None:
        """Add a classification annotation to the given object hypothesis.

        :param object_hypothesis: The object hypothesis to annotate.
        :param class_name: The name of the annotation class.
        """
        classification_annotation = Classification()
        classification_annotation.classname = class_name
        classification_annotation.source = "StaticObjectDetectorAnnotator"
        classification_annotation.confidence = 1.0
        object_hypothesis.annotations.append(classification_annotation)

    def get_rotation_list_based_on_parameters(
        self, rotation_x: float, rotation_y: float, rotation_z: float, rotation_w: float
    ) -> List[float]:
        """Return a quaternion based on the parametrization of the Annotator.

        :param rotation_x: rotation about x-axis
        :param rotation_y: rotation about y-axis
        :param rotation_z: rotation about z-axis
        :param rotation_w: rotation about w-axis
        :return: 4-dim list with Quaternion
        """
        if self.descriptor.parameters.pose_use_euler_angles:
            rotation_matrix = get_rotation_matrix_from_euler_angles(
                rotation_x, rotation_y, rotation_z
            )
            rotation_list = list(get_quaternion_from_rotation_matrix(rotation_matrix))
        else:
            # Interpret values directly as a quaternion
            rotation_list = [rotation_x, rotation_y, rotation_z, rotation_w]
        return rotation_list

    def get_cloud_from_2d_bounding_box_region_of_interest(
        self, color_in_red_green_blue: npt.NDArray
    ) -> o3d.geometry.PointCloud:
        """
        Build the point cloud of the depth measurements inside the configured bounding
        box.

        :param color_in_red_green_blue: The color image in RGB order, scaled to the depth image
            resolution.
        :return: The colored point cloud of the depth measurements inside the box.
        """
        parameters = self.descriptor.parameters
        box = Rectangle(
            position=Point2D(x=parameters.bounding_box_x, y=parameters.bounding_box_y),
            width=parameters.bounding_box_width,
            height=parameters.bounding_box_height,
        )
        left, top, right, bottom = (
            self.get_common_analysis_structure()
            .get(CommonAnalysisStructureViews.COLOR_TO_DEPTH_RATIO)
            .depth_region(box)
            .get_corner_points()
        )
        mask = np.zeros_like(self.depth, dtype=np.uint8)
        mask[top:bottom, left:right] = 255
        return get_cloud_from_color_depth_and_mask(
            color_in_red_green_blue,
            self.depth,
            mask,
            self.camera_intrinsics,
            mask_true_value=255,
        )
