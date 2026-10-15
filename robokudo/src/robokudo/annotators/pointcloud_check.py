"""
Point cloud validation and threshold checking.

This module provides an annotator for:

* Validating point cloud size
* Configurable threshold checking
* Customizable status responses
* Optional error raising

The annotator supports:

* Configurable point count threshold
* Different status returns for above/below threshold
* Optional exception raising on failure
* Detailed logging of check results

.. note::
   Point cloud must be available in CAS under CASViews.CLOUD.
"""

from __future__ import annotations

from dataclasses import dataclass


import open3d as o3d
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.exceptions import PointCloudThresholdError, PointCloudThresholdRelation


@dataclass(kw_only=True)
class PointcloudCheckAnnotator(BaseAnnotator):
    """
    Check if the CASViews.Cloud contains more than X points.

    .. warning::
       Will raise exception if point cloud not found in CAS.
    """

    point_threshold: int = 100
    """
    Decision boundary for: if CASViews.CLOUD has less than this amount of points.
    """

    status_below_threshold: Status = Status.FAILURE
    """
    Status when below threshold.
    """

    status_above_threshold: Status = Status.SUCCESS
    """
    Status when above threshold.
    """

    raise_on_failure: bool = True
    """
    Whether to raise exception on failure.
    """

    def compute(self) -> Status:
        """
        Check point cloud size against threshold.

        The method:

        * Loads point cloud from CAS
        * Counts number of points
        * Compares against threshold:
          * Below: Returns status_below_threshold
          * Above: Returns status_above_threshold
        * Optionally raises exceptions on failure
        * Logs detailed check results

        :return: Configured status based on point count
        :raises PointCloudThresholdError: If point count fails threshold and raise_on_failure is True
        """
        cloud = self.get_cas().get(CASViews.CLOUD)
        assert isinstance(cloud, o3d.geometry.PointCloud)
        point_count = len(cloud.points)
        if point_count < self.point_threshold:
            if self.status_below_threshold == Status.FAILURE and self.raise_on_failure:
                raise PointCloudThresholdError(
                    point_count=point_count,
                    threshold=self.point_threshold,
                    relation=PointCloudThresholdRelation.BELOW,
                )

            self.robokudo_logger.info(
                f"Scene Pointcloud size({point_count}) is below "
                f"threshold of {self.point_threshold}"
            )
            return self.status_below_threshold
        else:
            if self.status_above_threshold == Status.FAILURE and self.raise_on_failure:
                raise PointCloudThresholdError(
                    point_count=point_count,
                    threshold=self.point_threshold,
                    relation=PointCloudThresholdRelation.ABOVE,
                )

            self.robokudo_logger.info(
                f"Scene Pointcloud size({point_count}) is above "
                f"threshold of {self.point_threshold}"
            )
            return self.status_above_threshold
