from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from semantic_digital_twin.exceptions import ModelLoadError

if TYPE_CHECKING:
    from transformers import Pipeline


@dataclass(frozen=True)
class DescriptionMatchResult:
    """Result of scoring a natural language description against a category.

    :Example:

    .. code-block:: python

        scorer = DescriptionCategoryScorer()
        result = scorer.score("seating furniture", "A low fabric pouf in sapphire blue...")
        if result.is_match():
            print(f"Matched with score {result.score:.2f}")
    """

    category: str
    """
    The category phrase the description was scored against.
    """

    description: str
    """
    The natural language description that was evaluated.
    """

    score: float
    """
    Entailment probability in [0, 1] that the description belongs to the
    category.
    """

    def is_match(self, threshold: float = 0.5) -> bool:
        """
        Whether the score exceeds the given threshold.

        :param threshold: Minimum score to consider a match.
        :returns: True if score >= threshold.
        """
        return self.score >= threshold

    def __repr__(self) -> str:
        match_label = "MATCH" if self.is_match() else "NO MATCH"
        return (
            f"DescriptionMatchResult("
            f"category={self.category!r}, "
            f"score={self.score:.4f} [{match_label}])"
        )


@dataclass
class DescriptionCategoryScorer:
    """Scores how well natural language descriptions match a given category.

    Uses zero-shot NLI (Natural Language Inference) to estimate the probability
    that a description belongs to a category, without requiring any labelled data.

    .. warning::
        Pass **descriptive phrases**, not bare nouns, as category labels.
        The model treats the label as the hypothesis "This is a [category]",
        so ``"seating furniture"`` matches poufs, stools, and ottomans, while
        ``"chair"`` only matches objects that are literally called chairs.

    :Example:

    .. code-block:: python

        scorer = DescriptionCategoryScorer()

        # Use a descriptive phrase, not just a noun:
        pouf = "A low fabric pouf in sapphire blue, cylindrical shape, soft textured surface"
        result = scorer.score("seating furniture", pouf)
        print(result.score)        # 0.996 — correct match
        print(result.is_match())   # True

        # score("chair", pouf) would give 0.01 — too literal

        # Rank multiple concept phrases against one description:
        ranked = scorer.score_multiple(
            ["seating furniture", "storage furniture", "surface for placing objects"],
            pouf,
        )
        for r in ranked:
            print(r)
    """

    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    """
    HuggingFace model identifier for zero-shot classification.

    Must support the ``zero-shot-classification`` pipeline.
    """

    _pipeline: Pipeline = field(init=False, repr=False)
    """
    Loaded zero-shot classification pipeline.
    """

    def __post_init__(self) -> None:
        from transformers import pipeline

        try:
            self._pipeline = pipeline(
                "zero-shot-classification", model=self.model_name
            )
        except Exception as error:
            raise ModelLoadError(model_name=self.model_name) from error

    def score(self, category: str, description: str) -> DescriptionMatchResult:
        """
        Score how well a description matches a single category.

        The contrast label ``"not {category}"`` is included so the returned score
        is a meaningful probability rather than a trivially high value.

        .. note::
            Prefer descriptive phrases over single nouns.  ``"seating furniture"``
            correctly matches poufs and stools; ``"chair"`` does not.

        :param category: Concept phrase to match against (e.g. ``"seating furniture"``).
        :param description: The natural language description to evaluate.
        :returns: Match result with a score in [0, 1].
        """
        contrast = f"not {category}"
        output = self._pipeline(description, candidate_labels=[category, contrast])
        labels: list[str] = output["labels"]
        scores: list[float] = output["scores"]
        category_score: float = scores[labels.index(category)]
        return DescriptionMatchResult(
            category=category,
            description=description,
            score=category_score,
        )

    def score_multiple(
        self, categories: list[str], description: str
    ) -> list[DescriptionMatchResult]:
        """
        Score a description against multiple categories and return them ranked.

        All categories are evaluated in a single model call. The returned list is
        sorted by score descending, so the best-matching category comes first.

        .. note::
            Prefer descriptive phrases over single nouns for each category label.

        :param categories: Concept phrases to compare against
            (e.g. ``["seating furniture", "storage furniture", "surface for placing objects"]``).
        :param description: The natural language description to evaluate.
        :returns: List of match results, sorted by score descending.
        """
        output = self._pipeline(description, candidate_labels=categories)
        labels: list[str] = output["labels"]
        scores: list[float] = output["scores"]
        return [
            DescriptionMatchResult(
                category=label,
                description=description,
                score=score,
            )
            for label, score in zip(labels, scores)
        ]
