from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from semantic_digital_twin.semantic_annotations import description_matching
from semantic_digital_twin.semantic_annotations.description_matching import (
    DescriptionCategoryScorer,
    DescriptionMatch,
    PipelineTask,
    ZeroShotClassificationField,
    ZeroShotClassificationModel,
)


# %% a classification pipeline answering from a table
@dataclass
class TabulatedClassifier:
    """
    Stands in for a zero-shot classification pipeline, answering every description with
    fixed scores per category, and recording what it was asked.
    """

    scores_by_label: dict[str, float]
    """
    The score of every category, whatever the description.
    """

    calls: list[tuple[str, list[str]]] = field(default_factory=list)
    """
    The description and candidate labels of every call, in order.
    """

    loaded_models: list[str] = field(default_factory=list)
    """
    The models a pipeline was loaded for, in order.
    """

    def load(self, task: PipelineTask, model: str) -> TabulatedClassifier:
        """
        Stands in for :func:`transformers.pipeline`.
        """
        self.loaded_models.append(model)
        return self

    def __call__(self, description: str, candidate_labels: list[str]) -> dict:
        """
        Stands in for running the pipeline on *description*.
        """
        self.calls.append((description, candidate_labels))
        ranked = sorted(
            candidate_labels,
            key=lambda label: self.scores_by_label[label],
            reverse=True,
        )
        return {
            ZeroShotClassificationField.LABELS: ranked,
            ZeroShotClassificationField.SCORES: [
                self.scores_by_label[label] for label in ranked
            ],
        }


@pytest.fixture
def classifier(monkeypatch) -> TabulatedClassifier:
    tabulated = TabulatedClassifier(
        {
            "seating furniture": 0.9,
            "not seating furniture": 0.1,
            "storage furniture": 0.3,
            "surface for placing objects": 0.6,
        }
    )
    monkeypatch.setattr(description_matching, "pipeline", tabulated.load)
    return tabulated


# %% scoring
def test_a_description_is_scored_against_the_negation_of_its_category(
    classifier,
) -> None:
    match = DescriptionCategoryScorer().score("seating furniture", "A blue pouf")

    assert match == DescriptionMatch(
        category="seating furniture",
        description="A blue pouf",
        score=classifier.scores_by_label["seating furniture"],
    )
    assert classifier.calls == [
        ("A blue pouf", ["seating furniture", "not seating furniture"])
    ]


def test_several_categories_are_ranked_best_first(classifier) -> None:
    matches = DescriptionCategoryScorer().score_all(
        ["storage furniture", "seating furniture", "surface for placing objects"],
        "A blue pouf",
    )

    assert [match.category for match in matches] == [
        "seating furniture",
        "surface for placing objects",
        "storage furniture",
    ]


def test_the_model_is_loaded_once_on_first_use(classifier) -> None:
    scorer = DescriptionCategoryScorer()

    scorer.score("seating furniture", "A blue pouf")
    scorer.score("seating furniture", "A wooden stool")

    assert classifier.loaded_models == [
        ZeroShotClassificationModel.DEBERTA_V3_BASE_MNLI_FEVER_ANLI.value
    ]


# %% matches
@pytest.mark.parametrize("score, expected", [(0.49, False), (0.5, True), (0.9, True)])
def test_a_match_reaches_the_threshold(score: float, expected: bool) -> None:
    match = DescriptionMatch(category="c", description="d", score=score)

    assert match.is_match(threshold=0.5) is expected
