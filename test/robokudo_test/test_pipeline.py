from __future__ import annotations
import py_trees

from robokudo.annotators.core import BaseAnnotator
from robokudo.annotators.testing import (
    FakeCollectionReaderAnnotator,
    FailingAnnotator,
    SlowAnnotator,
)
from robokudo.pipeline import Pipeline
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from py_trees.composites import Sequence


def create_test_tree() -> Sequence:
    root = py_trees.composites.Sequence("Sequence", memory=True)

    # for annotator in [CollectionReader(), ThreadedAnnotator("ImagePreprocessor"), CaffeAnnotator("CaffeAnnotator")]:
    for annotator in [
        FakeCollectionReaderAnnotator(),
        FailingAnnotator("FailingAnnotator"),
        SlowAnnotator("CaffeAnnotator"),
    ]:
        root.add_child(annotator)

    return root


class TestPipeline(object):
    def test_find_common_analysis_structure_direct(self) -> None:
        root = Pipeline("Sequence")
        annotator1 = BaseAnnotator("1")
        annotator2 = BaseAnnotator("2")
        root.add_child(annotator1)
        root.add_child(annotator2)

        assert (
            root.common_analysis_structure == annotator1.get_common_analysis_structure()
        )

    def test_find_common_analysis_structure_from_deeper_leaf(self) -> None:
        root = Pipeline("Sequence")
        annotator1 = BaseAnnotator("1")
        annotator2 = BaseAnnotator("2")
        annotatorInSelector = BaseAnnotator("annotatorInSelector")

        selector = py_trees.composites.Selector("Selector", memory=True)
        success_after_two = py_trees.behaviours.Dummy(
            name="After Two",
        )
        always_running = py_trees.behaviours.Running(name="Running")
        selector.add_children([success_after_two, always_running, annotatorInSelector])

        root.add_child(annotator1)
        root.add_child(annotator2)
        root.add_child(selector)

        assert (
            root.common_analysis_structure
            == annotatorInSelector.get_common_analysis_structure()
        )

    def test_get_annotators_nested(self) -> None:
        root = Pipeline("Sequence")
        annotator1 = BaseAnnotator("1")
        annotator3 = BaseAnnotator("3")

        sub_sequence = py_trees.composites.Sequence("SubSequence", memory=True)
        annotator2 = BaseAnnotator("2")
        sub_sequence.add_child(annotator2)

        root.add_child(annotator1)
        root.add_child(sub_sequence)
        root.add_child(annotator3)

        assert root.get_annotators() == [annotator1, annotator2, annotator3]

    def test_pipeline_children1(self) -> None:
        root = Pipeline("Sequence")
        annotator1 = BaseAnnotator("1")
        annotator3 = BaseAnnotator("3")

        sub_sequence = py_trees.composites.Sequence("SubSequence", memory=True)
        annotator2 = BaseAnnotator("2")
        sub_sequence.add_child(annotator2)

        root.add_child(annotator1)
        root.add_child(sub_sequence)
        root.add_child(annotator3)

        assert root.pipeline_children() == [
            annotator1,
            annotator2,
            sub_sequence,
            annotator3,
        ]

    def test_pipeline_children2(self) -> None:
        root = Pipeline("Sequence")
        annotator1 = BaseAnnotator("1")
        annotator3 = BaseAnnotator("3")

        sub_pipeline = Pipeline("Subsequence")
        annotator2 = BaseAnnotator("2")
        sub_pipeline.add_child(annotator2)

        root.add_child(annotator1)
        root.add_child(sub_pipeline)
        root.add_child(annotator3)

        assert root.pipeline_children() == [annotator1, annotator3]

    def test_pipeline_children3(self) -> None:
        root = Pipeline("Sequence")
        annotator1 = BaseAnnotator("1")
        annotator3 = BaseAnnotator("3")

        # Pipeline children will not be fetched in nested RoboKudo Pipelines
        sub_pipeline = Pipeline("Subsequence")
        annotator2 = BaseAnnotator("2")
        sub_pipeline.add_child(annotator2)

        root.add_child(annotator1)
        root.add_child(annotator3)
        root.add_child(sub_pipeline)

        assert root.pipeline_children() == [annotator1, annotator3]
