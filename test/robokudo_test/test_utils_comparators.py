from __future__ import annotations

import numpy as np
import pytest
from robokudo.types.annotation import (
    BoundingBox3DAnnotation,
    Classification,
    ColorHistogram,
    SemanticColor,
    PositionAnnotation,
)
from robokudo.types.cv import Rect, ImageROI, Point2D
from robokudo.types.tf import Pose
from robokudo.utils.comparators import (
    TranslationComparator,
    BoundingBoxComparator,
    SizeComparator,
    HistogramComparator,
    SemanticColorComparator,
    AdditionalDataComparator,
    RegionOfInterestComparator,
    MaskComparator,
    ClassificationComparator,
    ImageRegionOfInterestComparator,
    OrientationComparator,
    PoseComparator,
)
from typing_extensions import TYPE_CHECKING, Callable, Any

if TYPE_CHECKING:
    import numpy.typing as npt


class TestUtilsComparators(object):

    @pytest.mark.parametrize(
        ["query_value", "object_value", "expected_similarity"],
        [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0),  # 100% the same
            (
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
            ),  # 0% the same (max_distance == distance)
            (
                (1.5, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
            ),  # 0% the same (distance > max_distance)
            (
                (0.5, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.5,
            ),  # 50% the same (max_distance / 2 == distance)
            (
                PositionAnnotation(translation=[0.0, 0.0, 0.0]),
                PositionAnnotation(translation=[0.0, 0.0, 0.0]),
                1.0,
            ),  # Compare PositionAnnotations (100% similarity)
            (
                PositionAnnotation(translation=[0.0, 0.0, 0.0]),
                (0.0, 0.0, 0.0),
                1.0,
            ),  # Compare PositionAnnotation with simple translation (100% similarity)
            (
                (0.0, 0.0, 0.0),
                PositionAnnotation(translation=[0.0, 0.0, 0.0]),
                1.0,
            ),  # Compare simple translation with PositionAnnotation (100% similarity)
            (
                PositionAnnotation(translation=[1.0, 0.0, 0.0]),
                PositionAnnotation(translation=[0.0, 0.0, 0.0]),
                0.0,
            ),  # Compare PositionAnnotations (0% similarity)
            (
                PositionAnnotation(translation=[1.0, 0.0, 0.0]),
                (0.0, 0.0, 0.0),
                0.0,
            ),  # Compare PositionAnnotation with simple translation (0% similarity)
            (
                (1.0, 0.0, 0.0),
                PositionAnnotation(translation=[0.0, 0.0, 0.0]),
                0.0,
            ),  # Compare simple translation with PositionAnnotation (0% similarity)
            (
                PositionAnnotation(translation=[0.5, 0.0, 0.0]),
                PositionAnnotation(translation=[0.0, 0.0, 0.0]),
                0.5,
            ),  # Compare PositionAnnotations (50% similarity)
            (
                PositionAnnotation(translation=[0.5, 0.0, 0.0]),
                (0.0, 0.0, 0.0),
                0.5,
            ),  # Compare PositionAnnotation with simple translation (50% similarity)
            (
                (0.5, 0.0, 0.0),
                PositionAnnotation(translation=[0.0, 0.0, 0.0]),
                0.5,
            ),  # Compare simple translation with PositionAnnotation (50% similarity)
        ],
    )
    def test_translation_comparator(
        self,
        query_value: tuple[float, float, float],
        object_value: tuple[float, float, float],
        expected_similarity: float,
    ) -> None:
        comparator = TranslationComparator(weight=1.0, max_distance=1.0)
        assert comparator.compute_similarity(
            query_value, object_value
        ) == pytest.approx(expected_similarity)

    @pytest.fixture
    def bounding_box_from_tuple(
        self,
    ) -> Callable[[tuple[float, float, float]], BoundingBox3DAnnotation]:
        def _make(lengths: tuple[float, float, float]) -> BoundingBox3DAnnotation:
            bounding_box = BoundingBox3DAnnotation()
            bounding_box.x_length, bounding_box.y_length, bounding_box.z_length = (
                lengths
            )
            return bounding_box

        return _make

    @pytest.mark.parametrize(
        ["query_value", "object_value", "expected_similarity"],
        [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0),
            ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 1.0),
            ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0), 0.5),
            ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), 0.25),
        ],
    )
    def test_bounding_box_comparator(
        self,
        query_value: tuple[float, float, float],
        object_value: tuple[float, float, float],
        expected_similarity: float,
        bounding_box_from_tuple: Callable[
            [tuple[float, float, float]], BoundingBox3DAnnotation
        ],
    ) -> None:
        query_bounding_box = bounding_box_from_tuple(query_value)
        object_bounding_box = bounding_box_from_tuple(object_value)
        comparator = BoundingBoxComparator(weight=1.0)

        similarity = comparator.compute_similarity(
            query_bounding_box, object_bounding_box
        )
        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)

    @pytest.mark.parametrize(
        ["query_value", "object_value", "expected_similarity"],
        [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0),
            ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1.0),
            ([1.0, 1.0, 1.0], [2.0, 1.0, 1.0], 0.5),
            ([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], 0.25),
        ],
    )
    def test_size_comparator(
        self,
        query_value: list[float],
        object_value: list[float],
        expected_similarity: float,
    ) -> None:
        comparator = SizeComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_value, object_value)
        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)

    @pytest.fixture
    def classification_annotation_from_classname(
        self,
    ) -> Callable[[str], Classification]:
        def _make(classname: str) -> Classification:
            ann = Classification()
            ann.classname = classname
            return ann

        return _make

    @pytest.mark.parametrize(
        ["query_value", "object_value", "expected_similarity"],
        [
            (np.array([0, 0, 0]), np.array([0, 0, 0]), pytest.approx(1.0)),
            (
                np.array([255, 255, 255]),
                np.array([0, 0, 0]),
                pytest.approx(0.0, abs=4e-2),
            ),
            (
                np.array([100, 100, 100]),
                np.array([100, 0, 0]),
                pytest.approx(0.444, abs=1e-3),
            ),
        ],
    )
    def test_histogram_comparator(
        self,
        query_value: np.ndarray,
        object_value: np.ndarray,
        expected_similarity: pytest.approx,
    ) -> None:
        comparator = HistogramComparator(weight=1.0)

        query_histogram, _ = np.histogram(query_value.ravel(), bins=256, range=[0, 256])
        object_histogram, _ = np.histogram(
            object_value.ravel(), bins=256, range=[0, 256]
        )

        query_color_histogram = ColorHistogram()
        query_color_histogram.hist = query_histogram.astype(np.float32)

        object_color_histogram = ColorHistogram()
        object_color_histogram.hist = object_histogram.astype(np.float32)

        similarity = comparator.compute_similarity(
            query_color_histogram, object_color_histogram
        )

        assert isinstance(similarity, float)
        assert similarity == expected_similarity

    @pytest.mark.parametrize(
        ["query_value", "object_value", "expected_similarity"],
        [
            (("white", 1.0), ("black", 1.0), 0.0),  # Different colors, full ratio
            (("white", 1.0), ("black", 0.0), 0.0),  # Different colors, different ratio
            (("white", 0.0), ("white", 1.0), 0.0),  # Same colors, different ratio
            (("white", 1.0), ("white", 1.0), 1.0),  # Same colors, same ratio
            (("white", 0.5), ("white", 1.0), 0.5),  # Same colors, insecure ratio
        ],
    )
    def test_semantic_color_comparator(
        self,
        query_value: tuple[str, float],
        object_value: tuple[str, float],
        expected_similarity: float,
    ) -> None:
        comparator = SemanticColorComparator(weight=1.0)

        query_color = SemanticColor()
        query_color.color = query_value[0]
        query_color.ratio = query_value[1]

        object_color = SemanticColor()
        object_color.color = object_value[0]
        object_color.ratio = object_value[1]

        similarity = comparator.compute_similarity(query_color, object_color)

        assert isinstance(similarity, float)
        assert similarity == expected_similarity

    @pytest.mark.parametrize(
        ["query_value", "object_value", "expected_similarity"],
        [
            # Test floats
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.5),
            (1.0, 0.5, 2 / 3),
            # Test ints
            (1, 1, 1.0),
            (1, 0, 0.5),
            # Test strings
            ("test1", "test1", 1.0),
            ("test1", "test2", 0.0),
        ],
    )
    def test_additional_data_comparator(
        self, query_value: Any, object_value: Any, expected_similarity: float
    ) -> None:
        comparator = AdditionalDataComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_value, object_value)

        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)

    @pytest.mark.parametrize(
        ["query_value", "object_value", "expected_similarity"],
        [
            ((0, 0, 1, 1), (0, 0, 1, 1), 1.0),  # Full overlap
            ((2, 2, 1, 1), (0, 0, 1, 1), 0.0),  # No overlap
            ((0, 0, 1, 1), (0, 0, 3, 3), 0.25),  # Partial overlap
        ],
    )
    def test_region_of_interest_comparator(
        self,
        query_value: tuple[int, int, int, int],
        object_value: tuple[int, int, int, int],
        expected_similarity: float,
    ) -> None:
        query_rectangle = Rect()
        query_rectangle.pos.x = query_value[0]
        query_rectangle.pos.y = query_value[1]
        query_rectangle.width = query_value[2]
        query_rectangle.height = query_value[3]

        object_rectangle = Rect()
        object_rectangle.pos.x = object_value[0]
        object_rectangle.pos.y = object_value[1]
        object_rectangle.width = object_value[2]
        object_rectangle.height = object_value[3]

        comparator = RegionOfInterestComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_rectangle, object_rectangle)

        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)

    @pytest.mark.parametrize(
        ["query_value", "object_value", "expected_similarity"],
        [
            (
                np.array([[0.0, 0.0], [0.0, 0.0]]),
                np.array([[0.0, 0.0], [0.0, 0.0]]),
                0.0,
            ),
            (
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                1.0,
            ),
            (
                np.array([[1.0, 1.0], [0.0, 0.0]]),
                np.array([[1.0, 0.0], [0.0, 0.0]]),
                0.5,
            ),
            (
                np.array([[1.0, 1.0], [0.0, 0.0]]),
                np.array([[1.0, 0.0], [1.0, 0.0]]),
                1.0 / 3.0,
            ),
        ],
    )
    def test_mask_comparator(
        self,
        query_value: npt.NDArray,
        object_value: npt.NDArray,
        expected_similarity: float,
    ) -> None:
        comparator = MaskComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_value, object_value)

        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)

    @pytest.mark.parametrize(
        ["query_value", "object_value", "expected_similarity"],
        [
            (
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=1.0
                ),
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=1.0
                ),
                1.0,
            ),
            (
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=1.0
                ),
                Classification(
                    classification_type="CLASS", classname="chair", confidence=1.0
                ),
                0.0,
            ),
            (
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=1.0
                ),
                Classification(
                    classification_type="INSTANCE", classname="table", confidence=1.0
                ),
                0.0,
            ),
            (
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=0.75
                ),
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=0.95
                ),
                0.8,
            ),
            (
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=0.0
                ),
                Classification(
                    classification_type="INSTANCE", classname="chair", confidence=1.0
                ),
                0.0,
            ),
        ],
    )
    def test_classification_comparator(
        self, query_value: Pose, object_value: Pose, expected_similarity: float
    ) -> None:
        comparator = ClassificationComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_value, object_value)

        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)

    @pytest.mark.parametrize(
        ["query_value", "object_value", "expected_similarity"],
        [
            (
                ImageROI(
                    roi=Rect(pos=Point2D(x=0, y=0), width=2, height=2),
                    mask=np.array([[1.0, 0.0], [0.0, 0.0]]),
                ),
                ImageROI(
                    roi=Rect(pos=Point2D(x=0, y=0), width=2, height=2),
                    mask=np.array([[1.0, 0.0], [0.0, 0.0]]),
                ),
                1.0,
            ),  # Positions and masks are identical
            (
                ImageROI(
                    roi=Rect(pos=Point2D(x=0, y=0), width=2, height=2),
                    mask=np.array([[1.0, 0.0], [0.0, 0.0]]),
                ),
                ImageROI(
                    roi=Rect(pos=Point2D(x=3, y=3), width=2, height=2),
                    mask=np.array([[0.0, 1.0], [0.0, 0.0]]),
                ),
                0.0,
            ),  # Roi and mask have no similarity
            (
                ImageROI(
                    roi=Rect(pos=Point2D(x=0, y=0), width=2, height=2),
                    mask=np.array([[1.0, 0.0], [0.0, 0.0]]),
                ),
                ImageROI(
                    roi=Rect(pos=Point2D(x=3, y=3), width=2, height=2),
                    mask=np.array([[1.0, 0.0], [0.0, 0.0]]),
                ),
                0.5,
            ),  # Masks are identical, position is not
            (
                ImageROI(
                    roi=Rect(pos=Point2D(x=0, y=0), width=2, height=2),
                    mask=np.array([[1.0, 0.0], [0.0, 0.0]]),
                ),
                ImageROI(
                    roi=Rect(pos=Point2D(x=0, y=0), width=2, height=2),
                    mask=np.array([[0.0, 1.0], [0.0, 0.0]]),
                ),
                0.5,
            ),  # Positions are identical, masks are not
            # Tests for full image masks
            (
                ImageROI(
                    roi=Rect(pos=Point2D(x=0, y=0), width=1, height=1),
                    mask=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                ),
                ImageROI(
                    roi=Rect(pos=Point2D(x=0, y=0), width=1, height=1),
                    mask=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                ),
                1.0,
            ),  # Positions and masks are identical
            (
                ImageROI(
                    roi=Rect(pos=Point2D(x=0, y=0), width=1, height=1),
                    mask=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                ),
                ImageROI(
                    roi=Rect(pos=Point2D(x=2, y=2), width=1, height=1),
                    mask=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                ),
                0.5,
            ),  # Positions are not identical, but roi masks are
        ],
    )
    def test_image_region_of_interest_comparator(
        self, query_value: Pose, object_value: Pose, expected_similarity: float
    ) -> None:
        comparator = ImageRegionOfInterestComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_value, object_value)

        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)

    @pytest.mark.parametrize(
        ["query_value", "object_value", "expected_similarity"],
        [
            (np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0]), 1.0),
            (np.array([0.0, 0.0, 0.0, -1.0]), np.array([0.0, 0.0, 0.0, 1.0]), 0.0),
            (
                np.array([0.7071068, 0.0, 0.7071068, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                0.5,
            ),
        ],
    )
    def test_orientation_comparator(
        self, query_value: Pose, object_value: Pose, expected_similarity: float
    ) -> None:
        comparator = OrientationComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_value, object_value)

        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)

    @pytest.mark.parametrize(
        ["query_value", "object_value", "expected_similarity"],
        [
            (
                Pose(
                    translation=[0.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                ),
                Pose(
                    translation=[0.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                ),
                1.0,
            ),
            (
                Pose(
                    translation=[1.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                ),
                Pose(
                    translation=[0.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                ),
                0.5,
            ),
            (
                Pose(
                    translation=[0.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                ),
                Pose(
                    translation=[0.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0, -1.0],
                ),
                0.5,
            ),
        ],
    )
    def test_pose_comparator(
        self, query_value: Pose, object_value: Pose, expected_similarity: float
    ) -> None:
        comparator = PoseComparator(weight=1.0)

        similarity = comparator.compute_similarity(query_value, object_value)

        assert isinstance(similarity, float)
        assert similarity == pytest.approx(expected_similarity)
