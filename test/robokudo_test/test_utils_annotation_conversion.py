from __future__ import annotations
import sys

import numpy as np
import pytest
import sensor_msgs.msg
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

from robokudo.common_analysis_structure import (
    CommonAnalysisStructure,
    CommonAnalysisStructureViews,
)
from robokudo.types.annotation import (
    PoseAnnotation,
    StampedPoseAnnotation,
    PositionAnnotation,
    SemanticColor,
    Classification,
    StampedPositionAnnotation,
    Shape,
    Cuboid,
    Cylinder,
    Sphere,
    LocationAnnotation,
)
from robokudo.types.core import Annotation
from robokudo.types.cv import BoundingBox3D
from robokudo.types.tf import StampedTransform
from robokudo.utils.annotation_conversion import (
    PoseAnnotationToStampedPoseAnnotationConverter,
    PositionAnnotationToStampedPoseAnnotationConverter,
    SemanticColorToObjectDesignatorConverter,
    ClassificationToObjectDesignatorConverter,
    StampedPoseToObjectDesignatorConverter,
    PoseToObjectDesignatorConverter,
    PositionToObjectDesignatorConverter,
    StampedPositionToObjectDesignatorConverter,
    BoundingBox3DForShapeSizeConverter,
    ShapeToObjectDesignatorConverter,
    CuboidToObjectDesignatorConverter,
    CylinderToObjectDesignatorConverter,
    SphereToObjectDesignatorConverter,
    LocationToObjectDesignatorConverter,
)
from robokudo_msgs.msg import ObjectDesignator, ShapeSize

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import (
    Box as SemanticDigitalTwinBox,
    Cylinder as SemanticDigitalTwinCylinder,
    Sphere as SemanticDigitalTwinSphere,
)
from semantic_digital_twin.world_description.world_entity import Region
from . import _assertions


class TestUtilsAnnotationConversion(object):
    @pytest.fixture
    def common_analysis_structure_with_transform(self) -> CommonAnalysisStructure:
        common_analysis_structure = CommonAnalysisStructure()

        # tf = StampedTransform()
        # tf.rotation = (-0.5, 0.5, -0.5, 0.5)
        # tf.translation = (0.5, 0.5, 0.5)
        # common_analysis_structure.set(CommonAnalysisStructureViews.VIEWPOINT_CAMERA_TO_WORLD, tf)
        common_analysis_structure.camera_to_world_transform = (
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                pos_x=0.5,
                pos_y=0.5,
                pos_z=0.5,
                quat_x=-0.5,
                quat_y=0.5,
                quat_z=-0.5,
                quat_w=0.5,
            )
        )

        kinect_camera_info = sensor_msgs.msg.CameraInfo()
        kinect_camera_info.header.frame_id = "some_weird_non_default_frame_id"
        kinect_camera_info.header.stamp.sec = np.random.randint(sys.maxsize)
        kinect_camera_info.header.stamp.nanosec = np.random.randint(sys.maxsize)
        kinect_camera_info.width = 1024
        kinect_camera_info.height = 1280
        kinect_camera_info.k[0] = 1050.0
        kinect_camera_info.k[2] = 1050.0
        kinect_camera_info.k[4] = 639.5
        kinect_camera_info.k[5] = 479.5
        common_analysis_structure.set(
            CommonAnalysisStructureViews.CAMERA_INFO, kinect_camera_info
        )
        return common_analysis_structure

    def test_pose_annotation_to_stamped_pose_annotation_can_convert(self) -> None:
        converter = PoseAnnotationToStampedPoseAnnotationConverter()

        pose_annotation = PoseAnnotation()
        position_annotation = PositionAnnotation()

        assert converter.can_convert(pose_annotation, StampedPoseAnnotation) == True

        assert converter.can_convert(pose_annotation, PoseAnnotation) == False

        assert (
            converter.can_convert(position_annotation, StampedPoseAnnotation) == False
        )

    def test_pose_annotation_to_stamped_pose_annotation_convert(self) -> None:
        converter = PoseAnnotationToStampedPoseAnnotationConverter()

        pose_annotation = PoseAnnotation()
        pose_annotation.source = "some_weird_no_default_source"
        pose_annotation.rotation = np.random.rand(4)
        pose_annotation.translation = np.random.rand(3)

        stamped_pose_annotation = converter.convert(pose_annotation)

        assert stamped_pose_annotation.source == pose_annotation.source
        assert np.all(stamped_pose_annotation.rotation == pose_annotation.rotation)
        assert np.all(
            stamped_pose_annotation.translation == pose_annotation.translation
        )

    def test_position_annotation_to_stamped_pose_annotation_can_convert(self) -> None:
        converter = PositionAnnotationToStampedPoseAnnotationConverter()

        pose_annotation = PoseAnnotation()
        position_annotation = PositionAnnotation()

        assert converter.can_convert(position_annotation, StampedPoseAnnotation) == True

        assert converter.can_convert(pose_annotation, StampedPoseAnnotation) == False

        assert converter.can_convert(position_annotation, PoseAnnotation) == False

    def test_position_annotation_to_stamped_pose_annotation_convert(self) -> None:
        converter = PositionAnnotationToStampedPoseAnnotationConverter()

        position_annotation = PositionAnnotation()
        position_annotation.source = "some_weird_no_default_source"
        position_annotation.translation = np.random.rand(3)

        stamped_pose_annotation = converter.convert(position_annotation)

        assert stamped_pose_annotation.source == position_annotation.source
        assert np.all(
            stamped_pose_annotation.translation == position_annotation.translation
        )

    def test_semantic_color_to_object_designator_converter_can_convert(self) -> None:
        converter = SemanticColorToObjectDesignatorConverter()
        assert converter.can_convert(SemanticColor()) == True
        assert converter.can_convert(Annotation()) == False

    def test_semantic_color_to_object_designator_converter_convert(self) -> None:
        common_analysis_structure = CommonAnalysisStructure()
        object_designator = ObjectDesignator()

        converter = SemanticColorToObjectDesignatorConverter()

        semantic_color = SemanticColor()
        semantic_color.color = "some_weird_non_default_color"

        converter.convert(semantic_color, common_analysis_structure, object_designator)

        assert len(object_designator.color) == 1
        assert object_designator.color[0] == semantic_color.color

    def test_classification_to_object_designator_converter_can_convert(self) -> None:
        converter = ClassificationToObjectDesignatorConverter()
        assert converter.can_convert(Classification()) == True
        assert converter.can_convert(Annotation()) == False

    def test_classification_to_object_designator_converter_convert(self) -> None:
        common_analysis_structure = CommonAnalysisStructure()
        object_designator = ObjectDesignator()

        converter = ClassificationToObjectDesignatorConverter()

        classification = Classification()
        classification.classname = "some_weird_non_default_classname"

        converter.convert(classification, common_analysis_structure, object_designator)

        assert object_designator.type == classification.classname

    def test_stamped_pose_to_object_designator_converter_can_convert(self) -> None:
        converter = StampedPoseToObjectDesignatorConverter()
        assert converter.can_convert(StampedPoseAnnotation()) == True
        assert converter.can_convert(Annotation()) == False

    def test_stamped_pose_to_object_designator_converter_convert(self) -> None:
        common_analysis_structure = CommonAnalysisStructure()
        object_designator = ObjectDesignator()

        converter = StampedPoseToObjectDesignatorConverter()

        stamped_pose_annotation = StampedPoseAnnotation()
        stamped_pose_annotation.source = "some_weird_non_default_source"
        stamped_pose_annotation.translation = np.random.rand(3)
        stamped_pose_annotation.rotation = np.random.rand(4)
        stamped_pose_annotation.frame = "some_weird_non_default_frame_id"
        stamped_pose_annotation.timestamp = np.random.randint(sys.maxsize)

        converter.convert(
            stamped_pose_annotation, common_analysis_structure, object_designator
        )

        assert len(object_designator.pose) == 1
        pose: PoseStamped = object_designator.pose[0]
        assert pose.header.stamp.sec == stamped_pose_annotation.timestamp
        assert pose.header.frame_id == stamped_pose_annotation.frame
        assert pose.pose.position.x == stamped_pose_annotation.translation[0]
        assert pose.pose.position.y == stamped_pose_annotation.translation[1]
        assert pose.pose.position.z == stamped_pose_annotation.translation[2]
        assert pose.pose.orientation.x == stamped_pose_annotation.rotation[0]
        assert pose.pose.orientation.y == stamped_pose_annotation.rotation[1]
        assert pose.pose.orientation.z == stamped_pose_annotation.rotation[2]
        assert pose.pose.orientation.w == stamped_pose_annotation.rotation[3]

    def test_pose_to_object_designator_converter_can_convert(self) -> None:
        converter = PoseToObjectDesignatorConverter()
        assert converter.can_convert(PoseAnnotation()) == True
        assert converter.can_convert(Annotation()) == False

    def test_pose_to_object_designator_converter_convert_in_camera(
        self, common_analysis_structure_with_transform: CommonAnalysisStructure
    ) -> None:
        # common_analysis_structure_with_transform.set(CommonAnalysisStructureViews.VIEWPOINT_CAMERA_TO_WORLD, None)
        common_analysis_structure_with_transform.camera_to_world_transform = None
        kinect_camera_info = common_analysis_structure_with_transform.get(
            CommonAnalysisStructureViews.CAMERA_INFO
        )

        object_designator = ObjectDesignator()

        converter = PoseToObjectDesignatorConverter()

        pose_annotation = PoseAnnotation()
        pose_annotation.source = "some_weird_non_default_source"
        pose_annotation.translation = np.random.rand(3)
        pose_annotation.rotation = np.random.rand(4)

        converter.convert(
            pose_annotation, common_analysis_structure_with_transform, object_designator
        )

        assert len(object_designator.pose) == 1
        pose: PoseStamped = object_designator.pose[0]
        assert pose.header.frame_id == kinect_camera_info.header.frame_id
        assert pose.header.stamp.sec == kinect_camera_info.header.stamp.sec
        assert (
            pose.header.stamp.nanosec == 0
        )  # TODO: Nanoseconds are ignored in RoboKudo?
        assert pose.pose.position.x == pose_annotation.translation[0]
        assert pose.pose.position.y == pose_annotation.translation[1]
        assert pose.pose.position.z == pose_annotation.translation[2]
        assert pose.pose.orientation.x == pose_annotation.rotation[0]
        assert pose.pose.orientation.y == pose_annotation.rotation[1]
        assert pose.pose.orientation.z == pose_annotation.rotation[2]
        assert pose.pose.orientation.w == pose_annotation.rotation[3]

    def test_pose_to_object_designator_converter_convert_in_world(
        self, common_analysis_structure_with_transform: CommonAnalysisStructure
    ) -> None:
        camera_to_world_quaternion = (
            common_analysis_structure_with_transform.camera_to_world_transform.quaternion.to_list()
        )
        kinect_camera_info = common_analysis_structure_with_transform.get(
            CommonAnalysisStructureViews.CAMERA_INFO
        )

        object_designator = ObjectDesignator()

        converter = PoseToObjectDesignatorConverter()

        pose_annotation = PoseAnnotation()
        pose_annotation.source = "some_weird_non_default_source"
        pose_annotation.translation = np.random.rand(3)
        pose_annotation.rotation = np.random.rand(4)

        converter.convert(
            pose_annotation, common_analysis_structure_with_transform, object_designator
        )

        new_rotation = (
            R.from_quat(camera_to_world_quaternion)
            * R.from_quat(pose_annotation.rotation)
        ).as_quat(True)

        assert len(object_designator.pose) == 1
        pose: PoseStamped = object_designator.pose[0]
        assert pose.header.frame_id == "map"
        assert pose.header.stamp.sec == kinect_camera_info.header.stamp.sec
        assert (
            pose.header.stamp.nanosec == 0
        )  # TODO: Nanoseconds are ignored in RoboKudo?
        assert pose.pose.position.x == pose_annotation.translation[2] + 0.5
        assert pose.pose.position.y == -pose_annotation.translation[0] + 0.5
        assert pose.pose.position.z == -pose_annotation.translation[1] + 0.5
        assert np.isclose(pose.pose.orientation.x, new_rotation[0])
        assert np.isclose(pose.pose.orientation.y, new_rotation[1])
        assert np.isclose(pose.pose.orientation.z, new_rotation[2])
        assert np.isclose(pose.pose.orientation.w, new_rotation[3])

    def test_position_to_object_designator_converter_can_convert(self) -> None:
        converter = PositionToObjectDesignatorConverter()

        position_annotation = PositionAnnotation()
        other_ann = Annotation()

        assert converter.can_convert(position_annotation) == True
        assert converter.can_convert(other_ann) == False

    def test_position_to_object_designator_converter_convert(
        self, common_analysis_structure_with_transform: CommonAnalysisStructure
    ) -> None:
        camera_to_world_quaternion = (
            common_analysis_structure_with_transform.camera_to_world_transform.quaternion.to_list()
        )

        object_designator = ObjectDesignator()

        converter = PositionToObjectDesignatorConverter()

        position_annotation = PositionAnnotation()
        position_annotation.source = "some_weird_non_default_source"
        position_annotation.translation = np.random.rand(3)

        converter.convert(
            position_annotation,
            common_analysis_structure_with_transform,
            object_designator,
        )

        assert len(object_designator.pose) == 1
        pose: PoseStamped = object_designator.pose[0]
        assert pose.pose.position.x == position_annotation.translation[2] + 0.5
        assert pose.pose.position.y == -position_annotation.translation[0] + 0.5
        assert pose.pose.position.z == -position_annotation.translation[1] + 0.5
        _assertions.assert_quaternion_equal_up_to_sign(
            [
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ],
            camera_to_world_quaternion,
        )

    def test_stamped_position_to_object_designator_converter_can_convert(self) -> None:
        converter = StampedPositionToObjectDesignatorConverter()
        assert converter.can_convert(StampedPositionAnnotation()) == True
        assert converter.can_convert(Annotation()) == False

    def test_stamped_position_to_object_designator_converter_convert(
        self, common_analysis_structure_with_transform: CommonAnalysisStructure
    ) -> None:
        camera_to_world_quaternion = (
            common_analysis_structure_with_transform.camera_to_world_transform.quaternion.to_list()
        )
        object_designator = ObjectDesignator()

        converter = StampedPositionToObjectDesignatorConverter()

        stamped_pose_annotation = StampedPoseAnnotation()
        stamped_pose_annotation.source = "some_weird_non_default_source"
        stamped_pose_annotation.translation = np.random.rand(3)
        stamped_pose_annotation.rotation = np.random.rand(4)
        stamped_pose_annotation.frame = "some_weird_non_default_frame_id"
        stamped_pose_annotation.timestamp = np.random.randint(sys.maxsize)

        converter.convert(
            stamped_pose_annotation,
            common_analysis_structure_with_transform,
            object_designator,
        )

        assert len(object_designator.pose) == 1
        pose: PoseStamped = object_designator.pose[0]
        assert pose.header.stamp.sec == stamped_pose_annotation.timestamp
        assert pose.header.frame_id == stamped_pose_annotation.frame
        assert pose.pose.position.x == stamped_pose_annotation.translation[2] + 0.5
        assert pose.pose.position.y == -stamped_pose_annotation.translation[0] + 0.5
        assert pose.pose.position.z == -stamped_pose_annotation.translation[1] + 0.5
        _assertions.assert_quaternion_equal_up_to_sign(
            [
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ],
            camera_to_world_quaternion,
        )

    def test_bounding_box_3d_for_shape_size_converter_can_convert(self) -> None:
        converter = BoundingBox3DForShapeSizeConverter()
        assert converter.can_convert(BoundingBox3D()) == True
        assert converter.can_convert(Annotation()) == False

    def test_bounding_box_3d_for_shape_size_converter_convert(self) -> None:
        object_designator = ObjectDesignator()
        common_analysis_structure = CommonAnalysisStructure()

        converter = BoundingBox3DForShapeSizeConverter()

        bounding_box_3d = BoundingBox3D()
        bounding_box_3d.x_length = np.random.randint(sys.maxsize)
        bounding_box_3d.y_length = np.random.randint(sys.maxsize)
        bounding_box_3d.z_length = np.random.randint(sys.maxsize)

        converter.convert(bounding_box_3d, common_analysis_structure, object_designator)

        assert (
            object_designator.size == ""
        )  # TODO: Correct assertion when implementation of converter is complete
        assert len(object_designator.shape_size) == 1
        shape_size: ShapeSize = object_designator.shape_size[0]
        assert shape_size.dimensions.x == float(bounding_box_3d.x_length)
        assert shape_size.dimensions.y == float(bounding_box_3d.y_length)
        assert shape_size.dimensions.z == float(bounding_box_3d.z_length)
        assert shape_size.radius == 0.0

    def test_shape_to_object_designator_converter_can_convert(self) -> None:
        converter = ShapeToObjectDesignatorConverter()
        assert converter.can_convert(Shape()) == True
        assert converter.can_convert(Annotation()) == False

    def test_shape_to_object_designator_converter_convert(self) -> None:
        object_designator = ObjectDesignator()

        converter = ShapeToObjectDesignatorConverter()

        shape = Shape()
        shape.geometry = SemanticDigitalTwinSphere(radius=0.17)

        converter.convert(shape, CommonAnalysisStructure(), object_designator)

        assert len(object_designator.shape) == 1
        assert object_designator.shape[0] == shape.shape_name

    def test_cuboid_to_object_designator_converter_can_convert(self) -> None:
        converter = CuboidToObjectDesignatorConverter()
        assert converter.can_convert(Cuboid()) == True
        assert converter.can_convert(Annotation()) == False

    def test_cuboid_to_object_designator_converter_convert(self) -> None:
        object_designator = ObjectDesignator()

        converter = CuboidToObjectDesignatorConverter()

        shape = Cuboid()
        shape.geometry = SemanticDigitalTwinBox()

        converter.convert(shape, CommonAnalysisStructure(), object_designator)

        assert len(object_designator.shape) == 1
        assert object_designator.shape[0] == shape.shape_name

    def test_sphere_to_object_designator_converter_can_convert(self) -> None:
        converter = SphereToObjectDesignatorConverter()
        assert converter.can_convert(Sphere()) == True
        assert converter.can_convert(Annotation()) == False

    def test_cylinder_to_object_designator_converter_can_convert(self) -> None:
        converter = CylinderToObjectDesignatorConverter()
        assert converter.can_convert(Cylinder()) == True
        assert converter.can_convert(Annotation()) == False

    def test_cylinder_to_object_designator_converter_convert(self) -> None:
        object_designator = ObjectDesignator()

        converter = CylinderToObjectDesignatorConverter()

        shape = Cylinder()
        shape.geometry = SemanticDigitalTwinCylinder(width=0.22, height=0.41)

        converter.convert(shape, CommonAnalysisStructure(), object_designator)

        assert len(object_designator.shape) == 1
        assert object_designator.shape[0] == shape.shape_name

    def test_sphere_to_object_designator_converter_convert(self) -> None:
        object_designator = ObjectDesignator()

        converter = SphereToObjectDesignatorConverter()

        shape = Sphere()
        shape.geometry = SemanticDigitalTwinSphere(radius=np.random.random())

        converter.convert(shape, CommonAnalysisStructure(), object_designator)

        assert len(object_designator.shape) == 1
        assert object_designator.shape[0] == shape.shape_name

        assert len(object_designator.shape_size) == 1
        assert object_designator.shape_size[0].radius == shape.geometry.radius

    def test_location_to_object_designator_converter_can_convert(self) -> None:
        converter = LocationToObjectDesignatorConverter()
        assert converter.can_convert(LocationAnnotation()) == True
        assert converter.can_convert(Annotation()) == False

    def test_location_to_object_designator_converter_convert(self) -> None:
        object_designator = ObjectDesignator()

        converter = LocationToObjectDesignatorConverter()

        location = LocationAnnotation()
        location.region = Region(name=PrefixedName(name="some_weird_non_default_name"))

        converter.convert(location, CommonAnalysisStructure(), object_designator)

        assert object_designator.location == str(location.region.name)

    def test_location_to_object_designator_converter_convert_with_legacy_name_fallback(
        self,
    ) -> None:
        object_designator = ObjectDesignator()

        converter = LocationToObjectDesignatorConverter()

        location = LocationAnnotation()
        location.name = "legacy_location_name"

        converter.convert(location, CommonAnalysisStructure(), object_designator)

        assert object_designator.location == location.name
