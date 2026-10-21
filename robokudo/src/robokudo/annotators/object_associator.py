from py_trees.common import Status
from typing_extensions import List, Set, Optional

from robokudo.annotators.core import BaseAnnotator
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    ColorHistogram,
    Classification,
    PoseAnnotation,
)
from robokudo.types.cv import ImageRegionOfInterest
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.hypothesis_comparators import ObjectHypothesisComparator


class ObjectAssociator(BaseAnnotator):
    """
    An annotator that associates the objects of the current iteration with the objects
    of the last iteration.
    """

    def __init__(self, name: str = "ObjectAssociator") -> None:
        super().__init__(name)
        self.robokudo_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.tracked_objects: List[ObjectHypothesis] = []

        self.object_comparator = (
            ObjectHypothesisComparator(self.get_common_analysis_structure)
            .with_comparator_for(PoseAnnotation, 0.3, max_distance=0.5)
            .with_comparator_for(Classification, 0.3)
            .with_comparator_for(BoundingBox3DAnnotation, 0.1)
            .with_comparator_for(ColorHistogram, 0.2)
            .with_comparator_for(ImageRegionOfInterest, 0.1)
        )

    def update(self) -> Status:
        new_object_hypotheses: List[
            ObjectHypothesis
        ] = self.get_common_analysis_structure().filter_annotations_by_type(
            ObjectHypothesis
        )

        if len(self.tracked_objects) == 0:
            for new_object_hypothesis in new_object_hypotheses:
                self.tracked_objects.append(new_object_hypothesis)
            return Status.SUCCESS

        matched_new_ids: Set[int] = set()

        for new_object_hypothesis in new_object_hypotheses:
            best_match_id: Optional[int] = None
            best_similarity = 0.0

            for object_hypothesis_index, previous_object_hypothesis in enumerate(
                self.tracked_objects
            ):
                similarity = self.object_comparator.compute_similarity(
                    previous_object_hypothesis, new_object_hypothesis
                )
                if similarity > 0.5 and similarity > best_similarity:
                    best_match_id = object_hypothesis_index
                    best_similarity = similarity

            if best_match_id is not None and best_similarity > 0.5:
                self.robokudo_logger.info(
                    f"Associate object {new_object_hypothesis.id} with {best_match_id} ({self.tracked_objects[best_match_id].id})."
                )
                matched_new_ids.add(best_match_id)
                self.tracked_objects[best_match_id] = new_object_hypothesis
            else:
                self.tracked_objects.append(new_object_hypothesis)
                self.robokudo_logger.info(
                    f"Discovered new object {len(self.tracked_objects)}"
                )
        return Status.SUCCESS
