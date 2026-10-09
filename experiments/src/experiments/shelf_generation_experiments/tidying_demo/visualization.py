from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import rclpy
import trimesh
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker

from krrood.ormatic.utils import classproperty
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.world import World


# %% ROS node
@contextlib.contextmanager
def spinning_ros_node(name: str = "shelf_tidying_demo") -> Iterator[Node]:
    """
    Run a ROS node, spun in a background thread, for as long as the context lasts.

    :param name: The name of the node.
    :return: The running node.
    """
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node(name)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        yield node
    finally:
        executor.shutdown()
        thread.join()
        node.destroy_node()
        rclpy.shutdown()


# %% mesh resources
class MeshResourcePrefix(StrEnum):
    """
    The prefixes a marker's mesh resource starts with.
    """

    FILE = "file://"
    """
    A mesh on the local file system, which a browser cannot fetch.
    """

    GENERATED_FILE = "file:///tmp/"
    """
    A mesh written while a body was created, for example from a PLY file.
    """

    PACKAGE = "package://"
    """
    A mesh in the share directory of a ROS package, which ``foxglove_bridge`` serves
    over its websocket.
    """


@dataclass(eq=False)
class FoxgloveVizMarkerPublisher(VizMarkerPublisher):
    """
    A marker publisher whose meshes a browser-based Foxglove client can load.

    Generated meshes are converted to glTF and copied into the share directory of
    :attr:`mesh_resource_package`; meshes already installed in a ROS package are
    referred to by their ``package://`` resource.
    """

    mesh_resource_package: str = "foxglove_bridge"
    """
    The installed ROS package whose share directory converted meshes are copied into.
    """

    mesh_resource_directory: str = "shelf_generation_meshes"
    """
    Directory in the share directory of :attr:`mesh_resource_package` that converted
    meshes are copied into.
    """

    package_share_segment: str = "/share/"
    """
    The path segment separating an installation prefix from the name of the package
    whose share directory follows it.
    """

    @classproperty
    def gltf_up_axis_correction(cls) -> Rotation:
        """
        The rotation cancelling the Y-up to Z-up conversion Foxglove applies to every
        glTF mesh; the converted meshes are already Z-up.
        """
        return Rotation.from_euler("x", -90, degrees=True)

    def publish_markers(self) -> None:
        share_directory = (
            Path(get_package_share_directory(self.mesh_resource_package))
            / self.mesh_resource_directory
        )
        for marker in self.markers.markers:
            if marker.mesh_resource.startswith(MeshResourcePrefix.GENERATED_FILE):
                self._convert_generated_mesh(marker, share_directory)
            elif marker.mesh_resource.startswith(MeshResourcePrefix.FILE):
                self._refer_to_installed_mesh_by_package(marker)
        super().publish_markers()

    def _convert_generated_mesh(self, marker: Marker, share_directory: Path) -> None:
        """
        Convert the generated mesh of *marker* to glTF in *share_directory* and refer to
        it by its ``package://`` resource, which Foxglove can load with its texture.

        :param marker: The marker whose mesh is converted.
        :param share_directory: Directory converted meshes are copied into.
        """
        source_path = Path(marker.mesh_resource.removeprefix(MeshResourcePrefix.FILE))
        destination_directory = share_directory / source_path.parent.name
        destination_path = destination_directory / f"{source_path.stem}.glb"
        if not destination_path.exists():
            destination_directory.mkdir(parents=True, exist_ok=True)
            trimesh.load(source_path, force="mesh").export(
                destination_path, file_type="glb"
            )
        marker.mesh_resource = (
            f"{MeshResourcePrefix.PACKAGE}{self.mesh_resource_package}/"
            f"{self.mesh_resource_directory}/{source_path.parent.name}/"
            f"{destination_path.name}"
        )
        orientation = marker.pose.orientation
        corrected = (
            Rotation.from_quat(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            )
            * self.gltf_up_axis_correction
        )
        orientation.x, orientation.y, orientation.z, orientation.w = corrected.as_quat()

    def _refer_to_installed_mesh_by_package(self, marker: Marker) -> None:
        """
        Refer to the mesh of *marker*, installed in the share directory of a ROS package,
        by its ``package://`` resource; a mesh outside any share directory is left as it
        is.

        :param marker: The marker whose mesh resource is rewritten.
        """
        path = marker.mesh_resource.removeprefix(MeshResourcePrefix.FILE)
        _, separator, path_in_share = path.partition(self.package_share_segment)
        if not separator:
            return
        marker.mesh_resource = f"{MeshResourcePrefix.PACKAGE}{path_in_share}"


# %% viewers
class VisualizationBackend(StrEnum):
    """
    The viewer the markers of a world are published for.
    """

    FOXGLOVE = "foxglove"
    """
    A browser-based Foxglove client connected through ``foxglove_bridge``.
    """

    RVIZ = "rviz"
    """
    A local RViz2 instance, which loads meshes from the file system.
    """

    def publish(self, world: World, node: Node) -> VizMarkerPublisher:
        """
        Publish the markers of *world* for this viewer, replacing whatever an earlier
        run published, and keep publishing every change to *world*.

        :param world: The world to visualize.
        :param node: The node to publish with.
        :return: The publisher, which has to be kept alive.
        """
        publisher_type = (
            FoxgloveVizMarkerPublisher
            if self is VisualizationBackend.FOXGLOVE
            else VizMarkerPublisher
        )
        publisher = publisher_type(
            _world=world,
            node=node,
            qos_profile=QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL
            ),
        )
        publisher.markers.markers.insert(0, Marker(action=Marker.DELETEALL))
        publisher.publish_markers()
        return publisher
