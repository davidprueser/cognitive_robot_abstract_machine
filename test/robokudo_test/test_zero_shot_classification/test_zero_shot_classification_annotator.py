"""
Tests for routing zero-shot classifications of whole images and object hypotheses into
the CAS, independent of the model that classifies them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from PIL import Image
from typing_extensions import Any
from py_trees.common import Status

from robokudo.annotators.zero_shot_classification_annotator import (
    ZeroShotClassificationAnnotator,
)
from robokudo.cas import CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import Classification
from robokudo.types.scene import ObjectHypothesis

from ._color_scenes import (
    COLOR_BY_WORD,
    COLOR_VOCABULARY,
    hypothesis_classifications,
    object_hypothesis,
    rgb,
    solid_image,
    two_color_scene,
)

pytestmark = pytest.mark.usefixtures("annotator_output_buffer")

# %% channel mean classifier


@dataclass(kw_only=True)
class ChannelMeanClassificationAnnotator(ZeroShotClassificationAnnotator):
    """
    Ranks each color word of the vocabulary by how strongly the image shows the color it
    names.
    """

    def classify(self, images: list[Image.Image]) -> list[list[Classification]]:
        return [self._rank_colors(np.asarray(image)) for image in images]

    def _rank_colors(self, image: np.ndarray) -> list[Classification]:
        """
        :return: The kept classifications of one image, the most confident first.
        """
        mean_color = image.reshape(-1, 3).mean(axis=0) / 255.0
        classifications = [
            Classification(
                source=self.name,
                classname=word,
                confidence=float(mean_color @ (rgb(COLOR_BY_WORD[word]) / 255.0)),
            )
            for word in self.vocabulary
        ]
        classifications.sort(
            key=lambda classification: classification.confidence, reverse=True
        )
        return classifications[: self.save_top_k]


def _annotator_in_pipeline(
    color_image: np.ndarray, **configuration: Any
) -> ChannelMeanClassificationAnnotator:
    """
    :param configuration: Fields overriding classification against the color
        vocabulary.
    :return: An annotator whose CAS holds the given color image.
    """
    color_configuration = dict(
        vocabulary=COLOR_VOCABULARY, save_top_k=len(COLOR_VOCABULARY)
    )
    annotator = ChannelMeanClassificationAnnotator(
        **(color_configuration | configuration)
    )
    pipeline = Pipeline("Sequence")
    pipeline.add_child(annotator)
    pipeline.setup()
    annotator.get_cas().set(CASViews.COLOR_IMAGE, color_image)
    return annotator


def _scene_classifications(
    annotator: ZeroShotClassificationAnnotator,
) -> list[Classification]:
    return annotator.get_cas().filter_annotations_by_type(Classification)


def _classnames(classifications: list[Classification]) -> list[str]:
    return [classification.classname for classification in classifications]


# %% scene scope


def test_scene_scope_adds_the_image_classifications_to_the_cas() -> None:
    color = COLOR_BY_WORD["green"]
    annotator = _annotator_in_pipeline(solid_image(color))

    status = annotator.compute()

    assert status == Status.SUCCESS
    assert _classnames(_scene_classifications(annotator)) == _classnames(
        annotator.classify([Image.fromarray(solid_image(rgb(color)))])[0]
    )


def test_images_are_classified_in_rgb_channel_order() -> None:
    """
    The CAS color image is stored in BGR channel order; classifiers receive it in RGB.
    """
    annotator = _annotator_in_pipeline(solid_image(COLOR_BY_WORD["red"]))

    annotator.compute()

    assert _scene_classifications(annotator)[0].classname == "red"


# %% object hypothesis scope


def test_object_hypothesis_scope_classifies_each_hypothesis_by_its_region() -> None:
    annotator = _annotator_in_pipeline(
        two_color_scene(), analysis_scope=ObjectHypothesis
    )
    red_object = object_hypothesis(x=0, y=0, width=10, height=10)
    blue_object = object_hypothesis(x=30, y=0, width=10, height=10)
    annotator.get_cas().annotations.extend([red_object, blue_object])

    status = annotator.compute()

    assert status == Status.SUCCESS
    assert hypothesis_classifications(red_object)[0].classname == "red"
    assert hypothesis_classifications(blue_object)[0].classname == "blue"


def test_object_hypothesis_scope_skips_hypotheses_rejected_by_filter() -> None:
    rejected_object = object_hypothesis(x=0, y=0, width=10, height=10)
    annotator = _annotator_in_pipeline(
        two_color_scene(),
        analysis_scope=ObjectHypothesis,
        hypothesis_filter=lambda hypothesis: hypothesis is not rejected_object,
    )
    accepted_object = object_hypothesis(x=30, y=0, width=10, height=10)
    annotator.get_cas().annotations.extend([rejected_object, accepted_object])

    annotator.compute()

    assert hypothesis_classifications(rejected_object) == []
    assert len(hypothesis_classifications(accepted_object)) == (annotator.save_top_k)


def test_object_hypothesis_scope_without_hypotheses_adds_nothing() -> None:
    annotator = _annotator_in_pipeline(
        two_color_scene(), analysis_scope=ObjectHypothesis
    )

    status = annotator.compute()

    assert status == Status.SUCCESS
    assert annotator.get_cas().annotations == []


# %% unsupported scope


def test_unsupported_analysis_scope_fails() -> None:
    annotator = _annotator_in_pipeline(
        solid_image(COLOR_BY_WORD["red"]), analysis_scope=CASViews.DEPTH_IMAGE
    )

    assert annotator.compute() == Status.FAILURE
