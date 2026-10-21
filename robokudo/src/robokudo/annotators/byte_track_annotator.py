from __future__ import annotations

import copy

import numpy as np
import open3d as o3d
import supervision as sv
from py_trees.common import Status
from supervision.config import CLASS_NAME_DATA_FIELD
from trackers import ByteTrackTracker
from typing_extensions import TYPE_CHECKING, List

from robokudo.annotators.core import BaseAnnotator
from robokudo.common_analysis_structure import CommonAnalysisStructureViews
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis

if TYPE_CHECKING:
    import numpy.typing as npt


class ByteTrackAnnotator(BaseAnnotator):
    """A ByteTrack annotator that tracks detected objects across pipeline ticks.

    The annotator is capable of utilizing:

    * ObjectHypothesis region of interest
    * ObjectHypothesis region of interest + Mask
    * ObjectHypothesis region of interest + Mask + Classification annotation
    """

    def __init__(self, name: str = "ByteTrackAnnotator") -> None:
        super().__init__(name)

        self.tracker: ByteTrackTracker = ByteTrackTracker()
        """Main ByteTrack instance for object instance tracking."""

        self.last_timestamp: float = 0.0
        """Timestamp of the last data received, used for tracker reset detection."""

        self.label_annotator = sv.RichLabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
        """A special label annotator, that cannot simply be used like the annotators in `self.visualization_annotators`."""

        self.visualization_annotators = [
            sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK),
            sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX),
            sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK),
        ]
        """A list of simple supervision annotators to use for visualization. They will be applied in order."""

    def update(self) -> Status:
        common_analysis_structure = self.get_common_analysis_structure()

        data_timestamp = common_analysis_structure.get(
            CommonAnalysisStructureViews.DATA_TIMESTAMP
        )
        if data_timestamp < self.last_timestamp:
            # This is expected when looping data
            self.robokudo_logger.debug("Time moved backward, resetting tracker.")
            self.tracker.reset()
        self.last_timestamp = data_timestamp

        object_hypotheses: List[ObjectHypothesis] = (
            common_analysis_structure.filter_annotations_by_type(ObjectHypothesis)
        )
        if len(object_hypotheses) == 0:
            self.robokudo_logger.debug(
                f"{len(object_hypotheses)} object hypothesis found, nothing to do."
            )
            return Status.SUCCESS

        regions_of_interest_xyxy = np.array(
            [
                object_hypothesis.region_of_interest.rectangle.get_corner_points()
                for object_hypothesis in object_hypotheses
            ]
        )
        if len(regions_of_interest_xyxy) == 0:
            self.robokudo_logger.debug(
                f"No region of interest found in {len(object_hypotheses)} object hypothesis."
            )
            return Status.FAILURE

        detections = {
            "xyxy": regions_of_interest_xyxy,
            "confidence": np.ones(
                len(regions_of_interest_xyxy), dtype=float
            ),  # Default confidence
        }

        masks = [
            object_hypothesis.region_of_interest.mask
            for object_hypothesis in object_hypotheses
        ]
        masks = list(filter(lambda mask: mask is not None, masks))
        if len(masks) == len(regions_of_interest_xyxy):
            camera_intrinsic: o3d.cuda.pybind.camera.PinholeCameraIntrinsic = (
                common_analysis_structure.get(
                    CommonAnalysisStructureViews.CAMERA_INTRINSIC
                )
            )

            # Masks must be restored to full color image size
            restored_masks = []
            for region_of_interest, mask in zip(regions_of_interest_xyxy, masks):
                restored_mask = np.zeros(
                    (camera_intrinsic.height, camera_intrinsic.width), dtype=np.uint8
                )
                restored_mask[
                    region_of_interest[1] : region_of_interest[3],
                    region_of_interest[0] : region_of_interest[2],
                ] = mask
                restored_masks.append(restored_mask)
            detections["mask"] = np.array(restored_masks).astype(bool)

            self.robokudo_logger.debug(
                f"Using {len(restored_masks)} masks for object tracking."
            )

        classifications: List[Classification] = []
        for object_hypothesis in object_hypotheses:
            classification = common_analysis_structure.filter_by_type(
                Classification, object_hypothesis.annotations
            )
            if len(classification) == 0:
                break
            classifications.append(classification[0])
        if len(classifications) == len(regions_of_interest_xyxy):
            detections["class_id"] = np.zeros(len(object_hypotheses))
            detections["confidence"] = np.zeros(len(object_hypotheses))
            detections["data"][CLASS_NAME_DATA_FIELD] = np.full(
                len(object_hypotheses), "NONE", dtype=str
            )

            for i, (object_hypothesis, cls) in enumerate(
                zip(object_hypotheses, classifications)
            ):
                detections["class_id"][i] = cls.class_id
                detections["confidence"][i] = cls.confidence
                detections["data"][CLASS_NAME_DATA_FIELD] = cls.classname

            self.robokudo_logger.debug(
                f"Using {len(object_hypotheses)} class ids for object tracking."
            )

        detections = sv.Detections(**detections)
        tracked_detections = self.tracker.update(detections)

        self.visualize(tracked_detections)

        return Status.SUCCESS

    def visualize(self, tracked_detections: sv.Detections) -> None:
        """Visualize the tracked detections and set the resulting image in the annotator output struct.

        :param tracked_detections: The tracked detections to visualize.
        """
        visualization_image: npt.NDArray[np.uint8] = copy.deepcopy(
            self.get_common_analysis_structure().get(
                CommonAnalysisStructureViews.COLOR_IMAGE
            )
        )

        if (
            tracked_detections.tracker_id is not None
            and len(tracked_detections.tracker_id) > 0
        ):
            for annotator in self.visualization_annotators:
                visualization_image = annotator.annotate(
                    visualization_image, tracked_detections
                )

            labels = []
            for i, track_id in enumerate(tracked_detections.tracker_id):
                label = f"{track_id}: "
                if tracked_detections.class_id is not None:
                    label += f"{tracked_detections['class_name'][i]}"
                else:
                    label += "unknown"
                label += f" ({tracked_detections.confidence[i]:.2f})"

                labels.append(label)
            visualization_image = self.label_annotator.annotate(
                scene=visualization_image, detections=tracked_detections, labels=labels
            )

        self.get_annotator_output_structure().set_image(visualization_image)
