"""
This module provides conversion methods from Annotations to other types, such as ROS
Message types.

It's mainly used to in the result generation to fill the query result from the available
annotations, hence it requires a target ObjectDesignator to act on.
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod

from geometry_msgs.msg import Vector3, PoseStamped
from typing_extensions import TYPE_CHECKING, Type, Optional, TypeVar, Generic

from robokudo_msgs.msg import ShapeSize
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world
from robokudo.defs import PACKAGE_NAME
from robokudo.common_analysis_structure import CommonAnalysisStructureViews
from robokudo.types.annotation import (
    PoseAnnotation,
    StampedPoseAnnotation,
    Classification,
    PositionAnnotation,
    SemanticColor,
    LocationAnnotation,
    Sphere,
    Cuboid,
    Cylinder,
    Shape,
    StampedPositionAnnotation,
)
from robokudo.types.cv import BoundingBox3D
from robokudo.types.core import Annotation

if TYPE_CHECKING:
    from robokudo_msgs.msg import ObjectDesignator
    from robokudo.common_analysis_structure import CommonAnalysisStructure

T = TypeVar("T", bound=Annotation)
S = TypeVar("S", bound=Shape)


class AnnotationToAnnotationConverter(ABC, Generic[T]):
    """
    An abstract converter class for converting annotations to other annotations.
    """

    @abstractmethod
    def can_convert(self, annotation: Annotation, target_annotation_type: Type) -> bool:
        """
        Checks whether the converter can convert the given annotation to the given
        target annotation type.

        :param annotation: The annotation to check for conversion.
        :param target_annotation_type: The target annotation type to convert
            `annotation` to.
        :returns: Whether the converter can convert `annotation` to
            `target_annotation_type`.
        """
        pass

    @abstractmethod
    def convert(
        self,
        annotation: T,
        common_analysis_structure: Optional[CommonAnalysisStructure] = None,
    ) -> Annotation:
        """
        Converts the given annotation to the given target annotation type.

        Use `self.can_convert` to check whether the converter is able to convert the
        given annotation to the desired type.

        :param annotation: The annotation to convert.
        :param common_analysis_structure: The common analysis structure to use for
            conversion.
        :returns: The converted annotation.
        """
        pass


class AnnotationToObjectDesignatorConverter(ABC, Generic[T]):
    """
    An abstract converter class for converting annotations to object designators.
    """

    @abstractmethod
    def can_convert(self, annotation: Annotation) -> bool:
        """
        Checks whether the converter can convert the given annotation to an object
        designator.

        :param annotation: The annotation to check for conversion.
        :returns: Whether the converter can convert `annotation` to `ObjectDesignator`.
        """
        pass

    @abstractmethod
    def convert(
        self,
        annotation: T,
        common_analysis_structure: CommonAnalysisStructure,
        object_designator: ObjectDesignator,
    ) -> None:
        """
        Converts the data of the given annotation to data in the given object
        designator.

        Modifies the object designator in-place.         Use `self.can_convert` to check
        whether the converter is able to convert the given annotation to an object
        designator.

        :param annotation: The annotation to convert.
        :param common_analysis_structure: The common analysis structure to use for
            conversion.
        :param object_designator: The object designator to fill with the data of
            `annotation`
        """
        pass


# %% annotation to annotation conversion


class PoseAnnotationToStampedPoseAnnotationConverter(
    AnnotationToAnnotationConverter[PoseAnnotation]
):
    """
    Extended `AnnotationToAnnotationConverter` that converts a `PoseAnnotation` to a
    `StampedPoseAnnotation`.
    """

    def can_convert(self, annotation: Annotation, target_annotation_type: Type) -> bool:
        return (
            isinstance(annotation, PoseAnnotation)
            and target_annotation_type == StampedPoseAnnotation
        )

    def convert(
        self,
        annotation: PoseAnnotation,
        common_analysis_structure: Optional[CommonAnalysisStructure] = None,
    ) -> StampedPoseAnnotation:
        spa = StampedPoseAnnotation()
        spa.source = annotation.source
        spa.translation = annotation.translation
        spa.rotation = annotation.rotation
        spa.source = annotation.source
        return spa


class PositionAnnotationToStampedPoseAnnotationConverter(
    AnnotationToAnnotationConverter[PositionAnnotation]
):
    """
    Extended `AnnotationToAnnotationConverter` that converts a `Positionnnotation` to a
    `StampedPoseAnnotation`.
    """

    def can_convert(self, annotation: Annotation, target_annotation_type: Type) -> bool:
        return (
            isinstance(annotation, PositionAnnotation)
            and target_annotation_type == StampedPoseAnnotation
        )

    def convert(
        self,
        annotation: PositionAnnotation,
        common_analysis_structure: Optional[CommonAnalysisStructure] = None,
    ) -> StampedPoseAnnotation:
        spa = StampedPoseAnnotation()
        spa.source = annotation.source
        spa.translation = annotation.translation
        spa.rotation = [0, 0, 0, 1]
        return spa


# %% annotation to object designator conversion


class SemanticColorToObjectDesignatorConverter(
    AnnotationToObjectDesignatorConverter[SemanticColor]
):
    """
    Extended `AnnotationToAnnotationConverter` that converts a `SemanticColor`
    annotation to `ObjectDesignator` data.
    """

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, SemanticColor)

    def convert(
        self,
        annotation: SemanticColor,
        common_analysis_structure: CommonAnalysisStructure,
        object_designator: ObjectDesignator,
    ) -> None:
        object_designator.color.append(annotation.color)


class ClassificationToObjectDesignatorConverter(
    AnnotationToObjectDesignatorConverter[Classification]
):
    """
    Extended `AnnotationToAnnotationConverter` that converts a `Classification`
    annotation to `ObjectDesignator` data.
    """

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, Classification)

    def convert(
        self,
        annotation: Classification,
        common_analysis_structure: CommonAnalysisStructure,
        object_designator: ObjectDesignator,
    ) -> None:
        object_designator.type = annotation.classname


# class PoseBase2ODConverter(AnnotationToObjectDesignatorConverter):
#    def transform_camera_pose_to_world_pose(self, annotation: Annotation, common_analysis_structure: common analysis structure) -> PoseAnnotation:
#        return transform_pose_from_camera_to_world(common_analysis_structure, annotation)
#
#    def pose_stamped_from_pose_annotation(self, pose_annotation: PoseAnnotation) -> PoseStamped:
#        """
#        Convert a RoboKudo Pose Annotation to a ROS-style PoseStamped.
#        Be warned though, that we don't know anything about the header/frame, because the normal annotation
#        doesn't carry this information.
#
#        :param pose_annotation:
#        :return: geometry_msgs.msg.PoseStamped with only the position and rotation information from pose_annotation
#        """
#
#        ps = PoseStamped()
#        ps.pose.position.x = pose_annotation.translation[0]
#        ps.pose.position.y = pose_annotation.translation[1]
#        ps.pose.position.z = pose_annotation.translation[2]
#
#        ps.pose.orientation.x = pose_annotation.rotation[0]
#        ps.pose.orientation.y = pose_annotation.rotation[1]
#        ps.pose.orientation.z = pose_annotation.rotation[2]
#        ps.pose.orientation.w = pose_annotation.rotation[3]
#        return ps


class StampedPoseToObjectDesignatorConverter(
    AnnotationToObjectDesignatorConverter[StampedPoseAnnotation]
):
    """
    Extended `AnnotationToAnnotationConverter` that converts a `StampedPoseAnnotation`
    to `ObjectDesignator` data.
    """

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, StampedPoseAnnotation)

    def convert(
        self,
        annotation: StampedPoseAnnotation,
        common_analysis_structure: CommonAnalysisStructure,
        object_designator: ObjectDesignator,
    ) -> None:
        pose_stamped = PoseStamped()
        pose_stamped.pose.position.x = annotation.translation[0]
        pose_stamped.pose.position.y = annotation.translation[1]
        pose_stamped.pose.position.z = annotation.translation[2]

        pose_stamped.pose.orientation.x = annotation.rotation[0]
        pose_stamped.pose.orientation.y = annotation.rotation[1]
        pose_stamped.pose.orientation.z = annotation.rotation[2]
        pose_stamped.pose.orientation.w = annotation.rotation[3]

        pose_stamped.header.frame_id = annotation.frame
        pose_stamped.header.stamp.sec = annotation.timestamp

        object_designator.pose.append(pose_stamped)
        object_designator.pose_source.append(annotation.source)


class PoseToObjectDesignatorConverter(
    AnnotationToObjectDesignatorConverter[PoseAnnotation]
):
    """
    Extended `StampedPoseToObjectDesignatorConverter` that converts a `PoseAnnotation`
    to `ObjectDesignator` data.
    """

    def __init__(self) -> None:
        self.pose_converter = PoseAnnotationToStampedPoseAnnotationConverter()
        self.stamped_pose_converter = StampedPoseToObjectDesignatorConverter()

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, PoseAnnotation)

    def convert(
        self,
        annotation: PoseAnnotation,
        common_analysis_structure: CommonAnalysisStructure,
        object_designator: ObjectDesignator,
    ) -> None:
        """
        Converts the data of the given annotation to data in the given object
        designator.

        Use `self.can_convert` to check whether the converter is able to convert the
        given annotation to an object designator. Conversion requires a valid camera to
        world transform in the given common analysis structure.

        :param annotation: The annotation to convert.
        :param common_analysis_structure: The common analysis structure to use for
            conversion.
        :param object_designator: The object designator to fill with the data of
            `annotation`
        """
        use_camera_coordinates = (
            common_analysis_structure.camera_to_world_transform is None
        )
        if use_camera_coordinates:
            pose_annotation = annotation
        else:
            pose_annotation = transform_pose_from_camera_to_world(
                common_analysis_structure, annotation
            )

        # First, convert the PoseAnnotation to a StampedPose
        spa = self.pose_converter.convert(pose_annotation)
        # Fill the missing header information
        spa.timestamp = common_analysis_structure.get(
            CommonAnalysisStructureViews.CAMERA_INFO
        ).header.stamp.sec

        # For the frame, check if we should return camera or world coordinates based on the previous check
        if use_camera_coordinates:
            spa.frame = common_analysis_structure.get(
                CommonAnalysisStructureViews.CAMERA_INFO
            ).header.frame_id
        else:
            spa.frame = "map"

        # Second, do the actual conversion and place info in ObjectDesignator
        self.stamped_pose_converter.convert(
            spa, common_analysis_structure, object_designator
        )


class PositionToObjectDesignatorConverter(
    AnnotationToObjectDesignatorConverter[PositionAnnotation]
):
    """
    Extended `AnnotationToAnnotationConverter` that converts a `PositionAnnotation` to
    `ObjectDesignator` data.
    """

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, PositionAnnotation)

    def convert(
        self,
        annotation: PositionAnnotation,
        common_analysis_structure: CommonAnalysisStructure,
        object_designator: ObjectDesignator,
    ) -> None:
        """
        Converts the data of the given annotation to data in the given object
        designator.

        Use `self.can_convert` to check whether the converter is able to convert the
        given annotation to an object designator. Conversion requires a valid camera to
        world transform in the given common analysis structure.

        :param annotation: The annotation to convert.
        :param common_analysis_structure: The common analysis structure to use for
            conversion.
        :param object_designator: The object designator to fill with the data of
            `annotation`
        """
        pose_annotation = PoseAnnotation()
        pose_annotation.source = annotation.source
        pose_annotation.translation.insert(0, annotation.translation[0])
        pose_annotation.translation.insert(1, annotation.translation[1])
        pose_annotation.translation.insert(2, annotation.translation[2])

        pose_annotation.rotation.insert(0, 0)
        pose_annotation.rotation.insert(1, 0)
        pose_annotation.rotation.insert(2, 0)
        pose_annotation.rotation.insert(3, 1)

        pose_map = transform_pose_from_camera_to_world(
            common_analysis_structure, pose_annotation
        )

        # Must be a PoseStamped due to type specification in ros message
        pose_stamped = PoseStamped()
        pose_stamped.pose.position.x = pose_map.translation[0]
        pose_stamped.pose.position.y = pose_map.translation[1]
        pose_stamped.pose.position.z = pose_map.translation[2]

        pose_stamped.pose.orientation.x = pose_map.rotation[0]
        pose_stamped.pose.orientation.y = pose_map.rotation[1]
        pose_stamped.pose.orientation.z = pose_map.rotation[2]
        pose_stamped.pose.orientation.w = pose_map.rotation[3]

        # We assume that the pose annotation is in CAMERA coordinates
        pose_stamped.header = copy.deepcopy(
            common_analysis_structure.get(
                CommonAnalysisStructureViews.CAMERA_INFO
            ).header
        )
        pose_stamped.header.frame_id = "map"
        object_designator.pose.append(pose_stamped)

        object_designator.pose_source.append(annotation.source)


class StampedPositionToObjectDesignatorConverter(
    AnnotationToObjectDesignatorConverter[StampedPositionAnnotation]
):
    """
    Extended `AnnotationToAnnotationConverter` that converts a
    `StampedPositionAnnotation` to `ObjectDesignator` data.
    """

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, StampedPositionAnnotation)

    def convert(
        self,
        annotation: StampedPositionAnnotation,
        common_analysis_structure: CommonAnalysisStructure,
        object_designator: ObjectDesignator,
    ) -> None:
        """
        Converts the data of the given annotation to data in the given object
        designator.

        Use `self.can_convert` to check whether the converter is able to convert the
        given annotation to an object designator. Conversion requires a valid camera to
        world transform in the given common analysis structure.

        :param annotation: The annotation to convert.
        :param common_analysis_structure: The common analysis structure to use for
            conversion.
        :param object_designator: The object designator to fill with the data of
            `annotation`
        """
        # Create pose from position
        pose_annotation = PoseAnnotation()
        pose_annotation.source = annotation.source
        pose_annotation.translation.insert(0, annotation.translation[0])
        pose_annotation.translation.insert(1, annotation.translation[1])
        pose_annotation.translation.insert(2, annotation.translation[2])

        pose_annotation.rotation.insert(0, 0)
        pose_annotation.rotation.insert(1, 0)
        pose_annotation.rotation.insert(2, 0)
        pose_annotation.rotation.insert(3, 1)

        pose_map = transform_pose_from_camera_to_world(
            common_analysis_structure, pose_annotation
        )

        # Must be a PoseStamped due to type specification in ros message
        pose_stamped = PoseStamped()
        pose_stamped.pose.position.x = pose_map.translation[0]
        pose_stamped.pose.position.y = pose_map.translation[1]
        pose_stamped.pose.position.z = pose_map.translation[2]

        pose_stamped.pose.orientation.x = pose_map.rotation[0]
        pose_stamped.pose.orientation.y = pose_map.rotation[1]
        pose_stamped.pose.orientation.z = pose_map.rotation[2]
        pose_stamped.pose.orientation.w = pose_map.rotation[3]

        # Keep stamp data
        pose_stamped.header.frame_id = annotation.frame
        pose_stamped.header.stamp.sec = annotation.timestamp

        object_designator.pose.append(pose_stamped)
        object_designator.pose_source.append(annotation.source)


class BoundingBox3DForShapeSizeConverter(
    AnnotationToObjectDesignatorConverter[BoundingBox3D]
):
    """
    Extended `AnnotationToAnnotationConverter` that converts a
    `robokudo.types.cv.BoundingBox3D` to `ObjectDesignator` data.
    """

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, BoundingBox3D)

    def convert(
        self,
        annotation: BoundingBox3D,
        common_analysis_structure: CommonAnalysisStructure,
        object_designator: ObjectDesignator,
    ) -> None:
        # TODO A (new!) Annotator should infer semantic size.
        #  Find a good decision criteria for size. This should be a semantic label! (small, large etc.)
        object_designator.size = f""
        vector = Vector3()
        vector.x = float(annotation.x_length)
        vector.y = float(annotation.y_length)
        vector.z = float(annotation.z_length)
        size = ShapeSize()
        size.dimensions = vector
        size.radius = float(0.0)
        object_designator.shape_size.append(size)


class ShapeToObjectDesignatorConverter(AnnotationToObjectDesignatorConverter[Shape]):
    """
    Extended `AnnotationToAnnotationConverter` that converts a `Shape` annotation to
    `ObjectDesignator` data.
    """

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, Shape)

    def convert(
        self,
        annotation: Shape,
        common_analysis_structure: CommonAnalysisStructure,
        object_designator: ObjectDesignator,
    ) -> None:
        object_designator.shape.append(annotation.shape_name)


class CuboidToObjectDesignatorConverter(AnnotationToObjectDesignatorConverter[Cuboid]):
    """
    Extended `ShapeToObjectDesignatorConverter` that converts a `Cuboid` annotation to
    `ObjectDesignator` data.
    """

    def __init__(self) -> None:
        self.shape_converter = ShapeToObjectDesignatorConverter()

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, Cuboid)

    def convert(
        self,
        annotation: Cuboid,
        common_analysis_structure: CommonAnalysisStructure,
        object_designator: ObjectDesignator,
    ) -> None:
        self.shape_converter.convert(
            annotation, common_analysis_structure, object_designator
        )


class CylinderToObjectDesignatorConverter(
    AnnotationToObjectDesignatorConverter[Cylinder]
):
    """
    Extended `ShapeToObjectDesignatorConverter` that converts a `Cylinder` annotation to
    `ObjectDesignator` data.
    """

    def __init__(self) -> None:
        self.shape_converter = ShapeToObjectDesignatorConverter()

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, Cylinder)

    def convert(
        self,
        annotation: Cylinder,
        common_analysis_structure: CommonAnalysisStructure,
        object_designator: ObjectDesignator,
    ) -> None:
        self.shape_converter.convert(
            annotation, common_analysis_structure, object_designator
        )


class SphereToObjectDesignatorConverter(AnnotationToObjectDesignatorConverter[Sphere]):
    """
    Extended `ShapeToObjectDesignatorConverter` that converts a `Sphere` annotation to
    `ObjectDesignator` data.
    """

    def __init__(self) -> None:
        self.shape_converter = ShapeToObjectDesignatorConverter()

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, Sphere)

    def convert(
        self,
        annotation: Sphere,
        common_analysis_structure: CommonAnalysisStructure,
        object_designator: ObjectDesignator,
    ) -> None:
        size = ShapeSize(radius=annotation.geometry.radius)

        robokudo_logger = logging.getLogger(PACKAGE_NAME)
        robokudo_logger.info(
            f"The center point of the Sphere annotation is currently not converted to OD."
        )

        object_designator.shape_size.append(size)
        self.shape_converter.convert(
            annotation, common_analysis_structure, object_designator
        )


class LocationToObjectDesignatorConverter(
    AnnotationToObjectDesignatorConverter[LocationAnnotation]
):
    """
    Extended `ShapeToObjectDesignatorConverter` that converts a `LocationAnnotation` to
    `ObjectDesignator` data.
    """

    def can_convert(self, annotation: Annotation) -> bool:
        return isinstance(annotation, LocationAnnotation)

    def convert(
        self,
        annotation: LocationAnnotation,
        common_analysis_structure: CommonAnalysisStructure,
        object_designator: ObjectDesignator,
    ) -> None:
        if annotation.region is not None:
            object_designator.location = str(annotation.region.name)
            return
        object_designator.location = annotation.name
