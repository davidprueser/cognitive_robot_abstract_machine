from __future__ import annotations

from pathlib import Path

import pytest
import trimesh
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker, MarkerArray

from experiments.shelf_generation_experiments.tidying_demo import visualization
from experiments.shelf_generation_experiments.tidying_demo.visualization import (
    FoxgloveVizMarkerPublisher,
    MeshResourcePrefix,
    VisualizationBackend,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)


# %% fixtures
@pytest.fixture
def share_directory(tmp_path: Path, monkeypatch) -> Path:
    """
    The share directory converted meshes are copied into, standing in for the one of the
    installed ``foxglove_bridge``.
    """
    directory = tmp_path / "share"
    monkeypatch.setattr(
        visualization, "get_package_share_directory", lambda package: str(directory)
    )
    return directory


@pytest.fixture
def foxglove_publisher(
    rclpy_node, cylinder_bot_world, share_directory
) -> FoxgloveVizMarkerPublisher:
    publisher = FoxgloveVizMarkerPublisher(_world=cylinder_bot_world, node=rclpy_node)
    yield publisher
    publisher.stop()


def _publish_single_marker(publisher: VizMarkerPublisher, marker: Marker) -> Marker:
    publisher.markers = MarkerArray(markers=[marker])
    publisher.publish_markers()
    return publisher.markers.markers[0]


# %% mesh resources
def test_an_installed_mesh_is_referred_to_by_its_package(
    foxglove_publisher, share_directory
) -> None:
    marker = Marker(
        mesh_resource=f"{MeshResourcePrefix.FILE}/opt/ros/jazzy/share/robot_description/meshes/base.stl"
    )

    published = _publish_single_marker(foxglove_publisher, marker)

    assert (
        published.mesh_resource
        == f"{MeshResourcePrefix.PACKAGE}robot_description/meshes/base.stl"
    )


def test_a_mesh_outside_any_package_is_left_as_it_is(
    foxglove_publisher, share_directory
) -> None:
    resource = f"{MeshResourcePrefix.FILE}/home/user/meshes/base.stl"

    published = _publish_single_marker(
        foxglove_publisher, Marker(mesh_resource=resource)
    )

    assert published.mesh_resource == resource


def test_a_generated_mesh_is_converted_and_turned_upright(
    foxglove_publisher, share_directory, tmp_path: Path
) -> None:
    mesh_directory = tmp_path / "generated"
    mesh_directory.mkdir()
    trimesh.creation.box().export(mesh_directory / "box.obj")
    marker = Marker(mesh_resource=f"{MeshResourcePrefix.FILE}{mesh_directory}/box.obj")
    marker.pose.orientation.w = 1.0

    published = _publish_single_marker(foxglove_publisher, marker)

    converted_path = (
        share_directory / foxglove_publisher.mesh_resource_directory / "generated"
    ) / "box.glb"
    assert converted_path.exists()
    assert published.mesh_resource == (
        f"{MeshResourcePrefix.PACKAGE}{foxglove_publisher.mesh_resource_package}/"
        f"{foxglove_publisher.mesh_resource_directory}/generated/box.glb"
    )
    orientation = published.pose.orientation
    assert [
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ] == pytest.approx(FoxgloveVizMarkerPublisher.gltf_up_axis_correction.as_quat())


# %% viewers
@pytest.mark.parametrize(
    "backend, publisher_type",
    [
        (VisualizationBackend.FOXGLOVE, FoxgloveVizMarkerPublisher),
        (VisualizationBackend.RVIZ, VizMarkerPublisher),
    ],
)
def test_every_viewer_gets_a_publisher_of_its_own_kind(
    backend, publisher_type, rclpy_node, cylinder_bot_world, share_directory
) -> None:
    publisher = backend.publish(cylinder_bot_world, rclpy_node)

    assert type(publisher) is publisher_type
    publisher.stop()
