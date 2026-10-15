"""
ROI adjustment for RoboKudo.

This module provides an annotator for adjusting the size of Region of Interest (ROI)
boundaries for object hypotheses. It can grow or shrink ROIs and their associated masks
by a specified pixel offset.
"""

from __future__ import annotations

from dataclasses import dataclass


from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.cv_helper import adjust_image_roi, adjust_mask


@dataclass(kw_only=True)
class ROIAdjusterAnnotator(BaseAnnotator):
    """
    Annotator for adjusting ROI sizes of object hypotheses.

    This annotator can grow or shrink ROIs (Regions of Interest) on object hypotheses by
    a specified pixel offset. It handles both the ROI boundaries and their associated
    masks, ensuring proper adjustment of both.
    """

    offset_pixel: int = 20
    """
    Pixels to add/subtract from ROI sides (positive grows, negative shrinks)
    """

    analysis_scope: type = ObjectHypothesis
    """
    Type of annotations to process.
    """

    fill_value_mask: int = 0
    """
    Value to fill new mask areas when growing ROIs.
    """

    def compute(self) -> Status:
        """
        Update ROIs by applying the configured pixel offset.

        For each object hypothesis in the analysis scope:
        - Adjusts the ROI boundaries by the specified pixel offset
        - If a mask exists, adjusts it accordingly with the specified fill value

        :return: SUCCESS after adjusting all ROIs
        """
        color = self.get_cas().get(CASViews.COLOR_IMAGE)

        object_hypotheses = self.get_cas().filter_annotations_by_type(
            self.analysis_scope
        )
        for object_hypothesis in object_hypotheses:
            adjust_image_roi(color, object_hypothesis.roi, self.offset_pixel)

            if object_hypothesis.roi.mask is not None:
                object_hypothesis.roi.mask = adjust_mask(
                    object_hypothesis.roi.mask,
                    self.offset_pixel,
                    fill_value=self.fill_value_mask,
                )

        return Status.SUCCESS
