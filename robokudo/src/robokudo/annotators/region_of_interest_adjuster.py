"""region of interest adjustment for RoboKudo.

This module provides an annotator for adjusting the size of Region of Interest (region of interest)
boundaries for object hypotheses. It can grow or shrink regions of interest and their associated
masks by a specified pixel offset.
"""

from __future__ import annotations

from timeit import default_timer

from py_trees.common import Status
from typing_extensions import Type

from robokudo.annotators.core import BaseAnnotator
from robokudo.common_analysis_structure import CommonAnalysisStructureViews
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.cv_helper import adjust_image_region_of_interest, adjust_mask
from robokudo.utils.error_handling import catch_and_raise_to_blackboard


class RegionOfInterestAdjusterAnnotator(BaseAnnotator):
    """Annotator for adjusting region of interest sizes of object hypotheses.

    This annotator can grow or shrink regions of interest (Regions of Interest) on object
    hypotheses by a specified pixel offset. It handles both the region of interest boundaries
    and their associated masks, ensuring proper adjustment of both.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for region of interest adjustment."""

        class Parameters:
            """Parameters for configuring region of interest adjustment behavior."""

            def __init__(self) -> None:
                self.offset_pixel: int = 20
                """Pixels to add/subtract from region of interest sides (positive grows, negative shrinks)"""

                self.analysis_scope: Type = ObjectHypothesis
                """Type of annotations to process."""

                self.fill_value_mask: int = 0
                """Value to fill new mask areas when growing regions of interest"""

        # overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "RegionOfInterestAdjusterAnnotator",
        descriptor: RegionOfInterestAdjusterAnnotator.Descriptor | None = None,
    ) -> None:
        """Initialize the region of interest adjuster.

        :param name: Name of the annotator instance
        :param descriptor: Configuration descriptor
        """
        super().__init__(name, descriptor)

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        """Update regions of interest by applying the configured pixel offset.

        For each object hypothesis in the analysis scope:
        - Adjusts the region of interest boundaries by the specified pixel offset
        - If a mask exists, adjusts it accordingly with the specified fill value

        :return: SUCCESS after adjusting all regions of interest
        """
        start_timer = default_timer()
        color = self.get_common_analysis_structure().get(
            CommonAnalysisStructureViews.COLOR_IMAGE
        )

        object_hypotheses = (
            self.get_common_analysis_structure().filter_annotations_by_type(
                self.descriptor.parameters.analysis_scope
            )
        )
        for object_hypothesis in object_hypotheses:
            adjust_image_region_of_interest(
                color,
                object_hypothesis.region_of_interest,
                self.descriptor.parameters.offset_pixel,
            )

            if object_hypothesis.region_of_interest.mask is not None:
                object_hypothesis.region_of_interest.mask = adjust_mask(
                    object_hypothesis.region_of_interest.mask,
                    self.descriptor.parameters.offset_pixel,
                    fill_value=self.descriptor.parameters.fill_value_mask,
                )

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
