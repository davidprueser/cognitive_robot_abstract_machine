from __future__ import annotations
from py_trees.common import Status

from robokudo.annotators.filter import FilterAnnotator
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import Classification, SemanticColor
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from robokudo.types.core import Annotation


def _filter_annotator_in_pipeline() -> FilterAnnotator:
    pipeline = Pipeline("Sequence")
    annotator = FilterAnnotator()
    pipeline.add_child(annotator)
    return annotator


def test_filter_annotator_keeps_annotations_when_no_function_is_configured() -> None:
    annotator = _filter_annotator_in_pipeline()
    classification = Classification(classname="cup")
    color = SemanticColor(color="red")
    annotator.get_cas().annotations = [classification, color]

    status = annotator.update()

    assert status == Status.SUCCESS
    assert annotator.get_cas().annotations == [classification, color]


def test_filter_annotator_keeps_only_annotations_matching_configured_function() -> None:
    annotator = _filter_annotator_in_pipeline()
    classification = Classification(classname="cup")
    color = SemanticColor(color="red")
    annotator.get_cas().annotations = [classification, color]
    annotator.function = lambda annotation: isinstance(annotation, Classification)

    status = annotator.update()

    assert status == Status.SUCCESS
    assert annotator.get_cas().annotations == [classification]


def test_filter_annotator_passes_configured_function_args_and_kwargs() -> None:
    annotator = _filter_annotator_in_pipeline()
    cup = Classification(classname="cup", confidence=0.8)
    low_confidence_cup = Classification(classname="cup", confidence=0.2)
    bowl = Classification(classname="bowl", confidence=0.9)
    annotator.get_cas().annotations = [cup, low_confidence_cup, bowl]

    def matches_class_with_minimum_confidence(
        annotation: Annotation, classname: str, *, minimum_confidence: float
    ) -> bool:
        return (
            isinstance(annotation, Classification)
            and annotation.classname == classname
            and annotation.confidence >= minimum_confidence
        )

    annotator.function = matches_class_with_minimum_confidence
    annotator.arguments = ("cup",)
    annotator.keyword_arguments = {"minimum_confidence": 0.5}

    status = annotator.update()

    assert status == Status.SUCCESS
    assert annotator.get_cas().annotations == [cup]
