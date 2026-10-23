"""
Shared Open3D serialization helpers for common analysis structure codecs.
"""

from __future__ import annotations

import base64
import tempfile
from typing_extensions import Any

import open3d as o3d


def is_open3d_point_cloud(value: Any) -> bool:
    """
    Return ``True`` when ``value`` is an Open3D point cloud.
    """
    return isinstance(value, o3d.geometry.PointCloud)


def encode_open3d_point_cloud_to_base64_pcd(point_cloud: Any) -> str:
    """
    Encode an Open3D point cloud to a base64-encoded PCD payload.
    """
    temporary_file = tempfile.NamedTemporaryFile(suffix=".pcd")
    o3d.io.write_point_cloud(temporary_file.name, point_cloud)
    temporary_file.seek(0)
    payload = base64.b64encode(temporary_file.read()).decode("ascii")
    temporary_file.close()
    return payload


def decode_open3d_point_cloud_from_base64_pcd(payload: str) -> Any:
    """
    Decode a base64-encoded PCD payload into an Open3D point cloud.
    """
    temporary_file = tempfile.NamedTemporaryFile(suffix=".pcd")
    temporary_file.write(base64.b64decode(payload.encode("ascii")))
    temporary_file.flush()
    point_cloud = o3d.io.read_point_cloud(temporary_file.name)
    temporary_file.close()
    return point_cloud
