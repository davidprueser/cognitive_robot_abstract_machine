from __future__ import annotations
import copy
import os
import uuid
from dataclasses import dataclass

import open3d as o3d
import pytest
import pymongo
import numpy as np

import robokudo.common_analysis_structure
from robokudo import world as rk_world
from robokudo.common_analysis_structure import (
    CommonAnalysisStructure,
    CommonAnalysisStructureViews,
)
from robokudo.descriptors.camera_configs.config_mongodb_playback import (
    MongoCameraConfig,
)
from robokudo.exceptions import StoredCameraTransformFrameMetadataMissing
from robokudo.io.storage import Storage
from robokudo.io.storage_reader_interface import StorageReaderInterface
from robokudo.types.cv import ImageROI
from robokudo.types.scene import ObjectHypothesis
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from pymongo.results import InsertOneResult
    from typing_extensions import Iterator


@dataclass
class CustomDataclassView:
    label: str
    confidence: float


pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="module temporarily disabled until storage functionality is migrated to ormatic",
)


@pytest.fixture(scope="module")
def storage_instance() -> Iterator[Storage]:
    db_name = "ONLY_UNITTESTS_test_db"
    storage = Storage(db_name)
    yield storage
    # Cleanup after tests
    storage.drop_database()


@pytest.fixture()
def common_analysis_structure_data() -> CommonAnalysisStructure:
    common_analysis_structure = CommonAnalysisStructure()
    """
    Mock common analysis structure data for testing.
    """
    common_analysis_structure.set(
        CommonAnalysisStructureViews.COLOR_IMAGE, np.random.rand(100, 100, 3)
    )
    common_analysis_structure.set(
        CommonAnalysisStructureViews.DEPTH_IMAGE, np.random.rand(100, 100)
    )

    return common_analysis_structure


def store_common_analysis_structure_in_storage(
    storage_instance: Storage, common_analysis_structure_data: CommonAnalysisStructure
) -> InsertOneResult:
    flat_common_analysis_structure = (
        storage_instance.generate_dict_from_real_common_analysis_structure(
            common_analysis_structure_data
        )
    )
    flat_common_analysis_structure["view_ids"] = {}

    # step 1: persist each view
    storage_instance.store_views_in_mongo(flat_common_analysis_structure)

    # step 2: persist common_analysis_structure and pointers to each view
    del flat_common_analysis_structure["views"]
    return storage_instance.store_common_analysis_structure_dict(
        flat_common_analysis_structure
    )


def create_point_cloud() -> o3d.geometry.PointCloud:
    """
    Create an Open3D point cloud with some sample data.
    """
    # Create a numpy array of points
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # Create the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud


def create_camera_intrinsic() -> o3d.camera.PinholeCameraIntrinsic:
    """
    Create an Open3D pinhole camera intrinsic for testing.
    """
    return o3d.camera.PinholeCameraIntrinsic(640, 480, 525.0, 526.0, 319.5, 239.5)


class TestStorage:
    def test_drop_database(self, storage_instance: Storage) -> None:
        db_name = storage_instance.db_name
        storage_instance.drop_database()

        # Verify the database no longer exists

        # Fetch environment variables which might have been used to configure
        # another MongoDB Host and Port
        # This was originally introduced to support unit tests.
        mongo_host = os.getenv("RK_MONGO_HOST", "localhost")
        mongo_port = int(os.getenv("RK_MONGO_PORT", 27017))
        client = pymongo.MongoClient(host=mongo_host, port=mongo_port)
        assert db_name not in client.list_database_names()

    def test_store_and_retrieve_common_analysis_structure(
        self,
        storage_instance: Storage,
        common_analysis_structure_data: CommonAnalysisStructure,
    ) -> None:
        result = store_common_analysis_structure_in_storage(
            storage_instance, common_analysis_structure_data
        )
        # Generate a dict from the common analysis structure and store it
        flat_common_analysis_structure = (
            storage_instance.generate_dict_from_real_common_analysis_structure(
                common_analysis_structure_data
            )
        )
        flat_common_analysis_structure["view_ids"] = {}

        # step 1: persist each view
        storage_instance.store_views_in_mongo(flat_common_analysis_structure)

        # step 2: persist common_analysis_structure and pointers to each view
        del flat_common_analysis_structure["views"]
        result = storage_instance.store_common_analysis_structure_dict(
            flat_common_analysis_structure
        )

        assert result.acknowledged

        # Retrieve the stored common analysis structure
        retrieved_common_analysis_structure = storage_instance.db.cas.find_one(
            {"_id": result.inserted_id}
        )
        assert retrieved_common_analysis_structure is not None
        assert (
            retrieved_common_analysis_structure["timestamp"]
            == common_analysis_structure_data.timestamp
        )
        assert (
            retrieved_common_analysis_structure["timestamp_readable"]
            == common_analysis_structure_data.timestamp_readable
        )

    def test_store_and_retrieve_common_analysis_structure_with_single_oh(
        self,
        storage_instance: Storage,
        common_analysis_structure_data: CommonAnalysisStructure,
    ) -> None:
        # Generate a dict from the common analysis structure and store it
        oh = ObjectHypothesis()
        oh.source = "test_storage"
        oh.id = 1
        oh.roi = ImageROI()
        oh.roi.roi.pos.x = 1
        oh.roi.roi.pos.y = 2
        oh.roi.roi.width = 20
        oh.roi.roi.height = 40
        oh.points = create_point_cloud()

        common_analysis_structure_data.annotations.append(oh)

        result = store_common_analysis_structure_in_storage(
            storage_instance, common_analysis_structure_data
        )

        flat_common_analysis_structure = (
            storage_instance.generate_dict_from_real_common_analysis_structure(
                common_analysis_structure_data
            )
        )
        flat_common_analysis_structure["view_ids"] = {}

        # step 1: persist each view
        storage_instance.store_views_in_mongo(flat_common_analysis_structure)

        # step 2: persist common_analysis_structure and pointers to each view
        del flat_common_analysis_structure["views"]
        result = storage_instance.store_common_analysis_structure_dict(
            flat_common_analysis_structure
        )

        assert result.acknowledged

        # Retrieve the stored common analysis structure
        retrieved_common_analysis_structure_record = storage_instance.db.cas.find_one(
            {"_id": result.inserted_id}
        )
        assert retrieved_common_analysis_structure_record is not None
        assert (
            retrieved_common_analysis_structure_record["timestamp"]
            == common_analysis_structure_data.timestamp
        )
        assert (
            retrieved_common_analysis_structure_record["timestamp_readable"]
            == common_analysis_structure_data.timestamp_readable
        )

        retrieved_common_analysis_structure = (
            robokudo.common_analysis_structure.CommonAnalysisStructure()
        )
        retrieved_common_analysis_structure_record["views"] = {}
        storage_instance.load_views_from_mongo_in_common_analysis_structure(
            retrieved_common_analysis_structure_record
        )

        # Bring flat common analysis structure representation into the proper common analysis structure class
        for view_name, view_content in retrieved_common_analysis_structure_record[
            "views"
        ].items():
            retrieved_common_analysis_structure.set(view_name, view_content)

        storage_instance.load_annotations_from_mongo_in_common_analysis_structure(
            retrieved_common_analysis_structure_record,
            retrieved_common_analysis_structure,
        )

        # Compare the most important views: COLOR and DEPTH Image
        assert np.array_equal(
            common_analysis_structure_data.get(
                CommonAnalysisStructureViews.COLOR_IMAGE
            ),
            retrieved_common_analysis_structure.get(
                CommonAnalysisStructureViews.COLOR_IMAGE
            ),
        )
        assert np.array_equal(
            common_analysis_structure_data.get(
                CommonAnalysisStructureViews.DEPTH_IMAGE
            ),
            retrieved_common_analysis_structure.get(
                CommonAnalysisStructureViews.DEPTH_IMAGE
            ),
        )

        assert len(retrieved_common_analysis_structure.annotations) == 1
        retrieved_oh = retrieved_common_analysis_structure.annotations[0]

        assert oh.source == retrieved_oh.source
        assert oh.id == retrieved_oh.id
        assert oh.roi.roi.pos.x == retrieved_oh.roi.roi.pos.x
        assert oh.roi.roi.pos.y == retrieved_oh.roi.roi.pos.y
        assert oh.roi.roi.width == retrieved_oh.roi.roi.width
        assert oh.roi.roi.height == retrieved_oh.roi.roi.height

        # Compare PointCloud
        test_pcd = create_point_cloud()

        np.testing.assert_array_equal(
            np.asarray(test_pcd.points), np.asarray(retrieved_oh.points.points)
        )

    def test_annotations_are_serialized_with_krrood(
        self,
        storage_instance: Storage,
        common_analysis_structure_data: CommonAnalysisStructure,
    ) -> None:
        oh = ObjectHypothesis()
        oh.source = "test_storage"
        oh.id = 1
        common_analysis_structure_data.annotations.append(oh)

        result = store_common_analysis_structure_in_storage(
            storage_instance, common_analysis_structure_data
        )
        assert result.acknowledged

        retrieved_common_analysis_structure_record = storage_instance.db.cas.find_one(
            {"_id": result.inserted_id}
        )
        assert retrieved_common_analysis_structure_record is not None
        assert (
            retrieved_common_analysis_structure_record["annotations_format"]
            == "krrood_v1"
        )
        assert isinstance(
            retrieved_common_analysis_structure_record["annotations"], list
        )
        assert (
            "__json_type__"
            in retrieved_common_analysis_structure_record["annotations"][0]
        )

    def test_store_and_retrieve_dynamic_extra_views(
        self,
        storage_instance: Storage,
        common_analysis_structure_data: CommonAnalysisStructure,
    ) -> None:
        common_analysis_structure_data.set(
            "custom_json", {"labels": ["cup", "plate"], "valid": True}
        )
        common_analysis_structure_data.set("custom_tuple", (1.2, 3.4))
        common_analysis_structure_data.set(
            "custom_numpy", np.array([[1, 2], [3, 4]], dtype=np.int16)
        )
        common_analysis_structure_data.set("custom_cloud", create_point_cloud())
        common_analysis_structure_data.set(
            "pc_cam_intrinsic", create_camera_intrinsic()
        )

        result = store_common_analysis_structure_in_storage(
            storage_instance, common_analysis_structure_data
        )
        assert result.acknowledged

        retrieved_common_analysis_structure_record = storage_instance.db.cas.find_one(
            {"_id": result.inserted_id}
        )
        assert retrieved_common_analysis_structure_record is not None
        assert "extra_views" not in retrieved_common_analysis_structure_record
        assert "view_ids" in retrieved_common_analysis_structure_record
        assert len(retrieved_common_analysis_structure_record["view_ids"]) >= 6

        retrieved_common_analysis_structure_record["views"] = {}
        storage_instance.load_views_from_mongo_in_common_analysis_structure(
            retrieved_common_analysis_structure_record
        )

        assert retrieved_common_analysis_structure_record["views"]["custom_json"] == {
            "labels": ["cup", "plate"],
            "valid": True,
        }
        assert retrieved_common_analysis_structure_record["views"]["custom_tuple"] == (
            1.2,
            3.4,
        )
        np.testing.assert_array_equal(
            retrieved_common_analysis_structure_record["views"]["custom_numpy"],
            np.array([[1, 2], [3, 4]], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            np.asarray(
                retrieved_common_analysis_structure_record["views"][
                    "custom_cloud"
                ].points
            ),
            np.asarray(create_point_cloud().points),
        )
        restored_intrinsic = retrieved_common_analysis_structure_record["views"][
            "pc_cam_intrinsic"
        ]
        expected_intrinsic = create_camera_intrinsic()
        assert restored_intrinsic.width == expected_intrinsic.width
        assert restored_intrinsic.height == expected_intrinsic.height
        np.testing.assert_allclose(
            np.asarray(restored_intrinsic.intrinsic_matrix),
            np.asarray(expected_intrinsic.intrinsic_matrix),
        )

    def test_extra_views_krrood_fallback_for_dataclass(
        self,
        storage_instance: Storage,
        common_analysis_structure_data: CommonAnalysisStructure,
    ) -> None:
        common_analysis_structure_data.set(
            "custom_dataclass", CustomDataclassView(label="cup", confidence=0.9)
        )

        result = store_common_analysis_structure_in_storage(
            storage_instance, common_analysis_structure_data
        )
        assert result.acknowledged

        retrieved_common_analysis_structure_record = storage_instance.db.cas.find_one(
            {"_id": result.inserted_id}
        )
        assert retrieved_common_analysis_structure_record is not None

        retrieved_common_analysis_structure_record["views"] = {}
        storage_instance.load_views_from_mongo_in_common_analysis_structure(
            retrieved_common_analysis_structure_record
        )

        restored = retrieved_common_analysis_structure_record["views"][
            "custom_dataclass"
        ]
        assert isinstance(restored, CustomDataclassView)
        assert restored.label == "cup"
        assert restored.confidence == 0.9

    def test_unknown_dynamic_view_type_raises(self) -> None:
        class UnsupportedView:
            pass

        common_analysis_structure = CommonAnalysisStructure()
        common_analysis_structure.set(
            CommonAnalysisStructureViews.COLOR_IMAGE, np.random.rand(10, 10, 3)
        )
        common_analysis_structure.set(
            CommonAnalysisStructureViews.DEPTH_IMAGE, np.random.rand(10, 10)
        )
        common_analysis_structure.set("custom_unsupported", UnsupportedView())

        with pytest.raises(TypeError):
            Storage.generate_dict_from_real_common_analysis_structure(
                common_analysis_structure
            )

    def test_camera_to_world_transform_roundtrip_preserves_world_references(
        self,
        storage_instance: Storage,
        common_analysis_structure_data: CommonAnalysisStructure,
    ) -> None:
        world_frame = f"map_{uuid.uuid4().hex[:8]}"
        camera_frame = f"camera_{uuid.uuid4().hex[:8]}"

        rk_world.init_world_with_entity_tracker()
        rk_world.setup_world_for_camera_frame(
            world_frame=world_frame, camera_frame=camera_frame
        )
        sem_world = rk_world.world_instance()
        camera_body = sem_world.get_body_by_name(camera_frame)
        world_body = sem_world.get_body_by_name(world_frame)

        transform = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.1,
            pos_y=0.2,
            pos_z=0.3,
            quat_x=0.0,
            quat_y=0.0,
            quat_z=0.0,
            quat_w=1.0,
            child_frame=camera_body,
            reference_frame=world_body,
        )
        common_analysis_structure_data.camera_to_world_transform = transform

        result = store_common_analysis_structure_in_storage(
            storage_instance, common_analysis_structure_data
        )
        assert result.acknowledged

        retrieved_common_analysis_structure_record = storage_instance.db.cas.find_one(
            {"_id": result.inserted_id}
        )
        assert retrieved_common_analysis_structure_record is not None
        transform_view_document = storage_instance.db[
            Storage.VIEW_COLLECTION_NAME
        ].find_one(
            {
                "_id": retrieved_common_analysis_structure_record["view_ids"][
                    CommonAnalysisStructureViews.CAMERA_TO_WORLD_TRANSFORM
                ]
            }
        )
        assert transform_view_document["metadata"]["reference_frame_name"] == str(
            world_body.name
        )
        assert transform_view_document["metadata"]["child_frame_name"] == str(
            camera_body.name
        )

        retrieved_common_analysis_structure_record["views"] = {}
        storage_instance.load_views_from_mongo_in_common_analysis_structure(
            retrieved_common_analysis_structure_record
        )
        restored_transform = retrieved_common_analysis_structure_record["views"][
            CommonAnalysisStructureViews.CAMERA_TO_WORLD_TRANSFORM
        ]

        assert isinstance(restored_transform, HomogeneousTransformationMatrix)
        np.testing.assert_allclose(restored_transform.to_np(), transform.to_np())
        assert restored_transform.child_frame is not None
        assert restored_transform.reference_frame is not None
        assert str(restored_transform.child_frame.name) == str(camera_body.name)
        assert str(restored_transform.reference_frame.name) == str(world_body.name)

    def test_storage_reader_rebinds_camera_to_world_transform_to_running_world(
        self,
        storage_instance: Storage,
        common_analysis_structure_data: CommonAnalysisStructure,
    ) -> None:
        stored_world_frame = f"stored_map_{uuid.uuid4().hex[:8]}"
        stored_camera_frame = f"stored_camera_{uuid.uuid4().hex[:8]}"

        rk_world.init_world_with_entity_tracker()
        rk_world.setup_world_for_camera_frame(
            world_frame=stored_world_frame,
            camera_frame=stored_camera_frame,
        )
        stored_world = rk_world.world_instance()
        stored_world_body = stored_world.get_body_by_name(stored_world_frame)
        stored_camera_body = stored_world.get_body_by_name(stored_camera_frame)
        transform = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.4,
            pos_y=0.5,
            pos_z=0.6,
            quat_x=0.0,
            quat_y=0.0,
            quat_z=0.0,
            quat_w=1.0,
            child_frame=stored_camera_body,
            reference_frame=stored_world_body,
        )
        common_analysis_structure_data.camera_to_world_transform = transform

        result = store_common_analysis_structure_in_storage(
            storage_instance, common_analysis_structure_data
        )
        assert result.acknowledged
        retrieved_common_analysis_structure_record = storage_instance.db.cas.find_one(
            {"_id": result.inserted_id}
        )
        assert retrieved_common_analysis_structure_record is not None

        rk_world.init_world_with_entity_tracker()
        running_world = rk_world.world_instance()
        assert len(running_world.bodies) == 0

        reader = StorageReaderInterface(
            MongoCameraConfig(db_name=storage_instance.db_name)
        )
        retrieved_common_analysis_structure_record["views"] = {}
        storage_instance.load_views_from_mongo_in_common_analysis_structure(
            retrieved_common_analysis_structure_record,
            excluded_view_names={
                CommonAnalysisStructureViews.CAMERA_TO_WORLD_TRANSFORM
            },
        )
        reader._restore_camera_to_world_transform(
            retrieved_common_analysis_structure_record
        )

        restored_transform = retrieved_common_analysis_structure_record["views"][
            CommonAnalysisStructureViews.CAMERA_TO_WORLD_TRANSFORM
        ]
        assert isinstance(restored_transform, HomogeneousTransformationMatrix)
        np.testing.assert_allclose(restored_transform.to_np(), transform.to_np())
        assert restored_transform.reference_frame is running_world.get_body_by_name(
            stored_world_frame
        )
        assert restored_transform.child_frame is running_world.get_body_by_name(
            stored_camera_frame
        )
        assert len(running_world.get_bodies_by_name(stored_world_frame)) == 1
        assert len(running_world.get_bodies_by_name(stored_camera_frame)) == 1

    def test_storage_reader_raises_custom_error_when_camera_transform_frame_metadata_is_missing(
        self,
    ) -> None:
        reader = StorageReaderInterface.__new__(StorageReaderInterface)
        view_document = {
            "payload": {},
            "metadata": {},
        }

        with pytest.raises(StoredCameraTransformFrameMetadataMissing):
            reader._decode_stored_camera_to_world_transform(view_document, {})
