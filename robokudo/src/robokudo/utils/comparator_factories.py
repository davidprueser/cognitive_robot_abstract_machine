from __future__ import annotations

from typing_extensions import Dict, Optional, Type, Any

from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    Classification,
    ColorHistogram,
    PoseAnnotation,
    PositionAnnotation,
    SemanticColor,
    SIFTAnnotation,
    StampedPoseAnnotation,
    StampedPositionAnnotation,
    StampedTransformAnnotation,
)
from robokudo.types.core import Annotation
from robokudo.types.core import Type as RoboKudoType
from robokudo.types.cv import ImageROI, Rect
from robokudo.types.tf import (
    Pose,
    Position,
    StampedPose,
    StampedPosition,
    StampedTransform,
)
from robokudo.utils.comparators import (
    BoundingBoxComparator,
    ClassificationComparator,
    FeatureComparator,
    HistogramComparator,
    ImageRegionOfInterestComparator,
    PoseComparator,
    RegionOfInterestComparator,
    SemanticColorComparator,
    ScaleInvariantFeatureTransformComparator,
    TranslationComparator,
)


class FeatureComparatorFactory:
    """
    A factory for creating feature comparators.
    """

    annotator_comparators: Dict[Type[Annotation], Type[FeatureComparator]] = {
        BoundingBox3DAnnotation: BoundingBoxComparator,
        Classification: ClassificationComparator,
        ColorHistogram: HistogramComparator,
        SemanticColor: SemanticColorComparator,
        PositionAnnotation: TranslationComparator,
        StampedPositionAnnotation: TranslationComparator,
        PoseAnnotation: PoseComparator,
        StampedPoseAnnotation: PoseComparator,
        StampedTransformAnnotation: PoseComparator,
        SIFTAnnotation: ScaleInvariantFeatureTransformComparator,
    }
    """
    Mapping of annotation types to feature comparators.
    """

    type_comparators: Dict[Type[RoboKudoType], Type[FeatureComparator]] = {
        Rect: RegionOfInterestComparator,
        ImageROI: ImageRegionOfInterestComparator,
        Position: TranslationComparator,
        StampedPosition: TranslationComparator,
        Pose: PoseComparator,
        StampedPose: PoseComparator,
        StampedTransform: PoseComparator,
    }
    """
    Mapping of robokudo types to feature comparators.
    """

    @classmethod
    def for_annotation(
        cls, annotation: Type[Annotation], weight: float, **kwargs: Any
    ) -> Optional[FeatureComparator]:
        """
        Get a feature comparator for the given annotation type and assign it the given
        weight.

        :param annotation: The annotation type to get a feature comparator for.
        :param weight: The weight to assign the annotation type.
        :return: A feature comparator instance for the given annotation type or None if
            the type is not supported.
        """
        if annotation not in cls.annotator_comparators:
            return None
        return FeatureComparatorFactory.annotator_comparators[annotation](
            weight, **kwargs
        )

    @classmethod
    def for_type(
        cls, robokudo_type: Type[RoboKudoType], weight: float, **kwargs: Any
    ) -> Optional[FeatureComparator]:
        """
        Get a feature comparator for the given robokudo type and assign it the given
        weight.

        :param robokudo_type: The robokudo type to get a feature comparator for.
        :param weight: The weight to assign the annotation type.
        :return: A feature comparator instance for the given annotation type or None if
            the type is not supported.
        """
        if robokudo_type not in cls.type_comparators:
            return None
        return FeatureComparatorFactory.type_comparators[robokudo_type](
            weight, **kwargs
        )
