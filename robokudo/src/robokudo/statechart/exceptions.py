"""
Errors of RoboKudo's statechart nodes.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from robokudo.exceptions import RoboKudoError

if TYPE_CHECKING:
    from robokudo.statechart.nodes import PipelineNode

# %% one pipeline per context


@dataclass(kw_only=True)
class SecondPipelineError(RoboKudoError):
    """
    Raised when a second pipeline is set up in a context whose CAS another pipeline
    already fills.
    """

    registered_pipeline: PipelineNode
    """
    The pipeline that fills the CAS of the context.
    """

    second_pipeline: PipelineNode
    """
    The pipeline that was set up after it.
    """

    def error_message(self) -> str:
        return (
            f"Pipeline {self.second_pipeline.unique_name} cannot be set up, because "
            f"pipeline {self.registered_pipeline.unique_name} already fills the CAS of "
            f"its context."
        )

    def suggest_correction(self) -> str:
        return (
            "run one pipeline per statechart, and nest further annotators in that "
            "pipeline instead of in a pipeline of their own."
        )
