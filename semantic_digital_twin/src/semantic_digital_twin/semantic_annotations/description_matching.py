from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

from transformers import Pipeline, pipeline
from typing_extensions import Any, Self


# %% models
class PipelineTask(StrEnum):
    """
    The Hugging Face pipeline tasks descriptions are matched with.
    """

    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"
    """
    Classifying a text into categories given only by name.
    """


class ZeroShotClassificationModel(StrEnum):
    """
    Hugging Face models that classify text into categories they were not trained on, by
    natural language inference.
    """

    DEBERTA_V3_BASE_MNLI_FEVER_ANLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    """
    DeBERTa-v3-base, fine-tuned on MultiNLI, Fever-NLI and Adversarial-NLI.
    """


class ZeroShotClassificationField(StrEnum):
    """
    The fields of the result of a ``zero-shot-classification`` pipeline.
    """

    LABELS = "labels"
    """
    The candidate categories, most likely first.
    """

    SCORES = "scores"
    """
    The probability of every category, in the order of the labels.
    """


@dataclass
class ZeroShotClassification:
    """
    The result of classifying one text into candidate categories.
    """

    labels: list[str]
    """
    The candidate categories, most likely first.
    """

    scores: list[float]
    """
    The probability of every category of :attr:`labels`, in the same order.
    """

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        """
        :param data: The result of a ``zero-shot-classification`` pipeline.
        :return: The classification it describes.
        """
        return cls(
            labels=data[ZeroShotClassificationField.LABELS],
            scores=data[ZeroShotClassificationField.SCORES],
        )


# %% matching descriptions to categories
@dataclass
class DescriptionMatch:
    """
    How well a natural language description belongs to a category.
    """

    category: str
    """
    The category phrase the description was scored against.
    """

    description: str
    """
    The description that was scored.
    """

    score: float
    """
    The probability that the description belongs to :attr:`category`.
    """

    def is_match(self, threshold: float = 0.5) -> bool:
        """
        :param threshold: The lowest score that counts as a match.
        :return: Whether :attr:`score` reaches *threshold*.
        """
        return self.score >= threshold


@dataclass
class DescriptionCategoryScorer:
    """
    Scores how well natural language descriptions belong to categories, by zero-shot
    natural language inference, without any labelled data.

    .. warning::
        A category is read as the hypothesis "This is a <category>", so descriptive
        phrases work better than bare nouns: ``"seating furniture"`` matches a pouf or a
        stool, while ``"chair"`` matches only what is called a chair.
    """

    model: ZeroShotClassificationModel = (
        ZeroShotClassificationModel.DEBERTA_V3_BASE_MNLI_FEVER_ANLI
    )
    """
    The model the descriptions are classified with.
    """

    @cached_property
    def classifier(self) -> Pipeline:
        """
        The zero-shot classification pipeline of :attr:`model`, loaded on first use.
        """
        return pipeline(PipelineTask.ZERO_SHOT_CLASSIFICATION, model=self.model.value)

    def score(self, category: str, description: str) -> DescriptionMatch:
        """
        :param category: The category phrase, for example ``"seating furniture"``.
        :param description: The description to score.
        :return: How well *description* belongs to *category* rather than to its
            negation, ``"not <category>"``.
        """
        classification = self._classify(description, [category, f"not {category}"])
        return DescriptionMatch(
            category=category,
            description=description,
            score=classification.scores[classification.labels.index(category)],
        )

    def score_all(
        self, categories: list[str], description: str
    ) -> list[DescriptionMatch]:
        """
        :param categories: The category phrases to compare.
        :param description: The description to score.
        :return: How well *description* belongs to each of *categories*, best first.
        """
        classification = self._classify(description, categories)
        return [
            DescriptionMatch(category=label, description=description, score=score)
            for label, score in zip(classification.labels, classification.scores)
        ]

    def _classify(
        self, description: str, candidate_labels: list[str]
    ) -> ZeroShotClassification:
        """
        :param description: The text to classify.
        :param candidate_labels: The categories to classify it into.
        :return: The classification of *description*.
        """
        return ZeroShotClassification.from_json(
            self.classifier(description, candidate_labels=candidate_labels)
        )
