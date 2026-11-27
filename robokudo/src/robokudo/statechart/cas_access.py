"""
The context extension that holds the CAS the nodes of a RoboKudo statechart work on.
"""

from __future__ import annotations

# %% Imports

from dataclasses import dataclass, field

from cramph.context import ContextExtension
from typing_extensions import TYPE_CHECKING

from robokudo.cas import CAS
from robokudo.statechart.exceptions import SecondPipelineError

if TYPE_CHECKING:
    from robokudo.statechart.nodes import PipelineNode

# %% CAS access


@dataclass(kw_only=True)
class CASAccess(ContextExtension):
    """
    Holds the current CAS of a statechart, which its one pipeline replaces with a fresh
    one whenever it starts or resets.

    .. note:: The CAS lives in the process that ticks the statechart and is never
        serialized with it.
    """

    cas: CAS = field(default_factory=CAS)
    """
    The CAS the nodes of the statechart work on now.
    """

    started_cas_count: int = field(default=0, init=False)
    """
    How many fresh CAS :meth:`start_new_cas` put in; the latest one has this as its id.
    """

    pipeline: PipelineNode | None = field(default=None, init=False)
    """
    The pipeline that fills :attr:`cas`, once it was set up.
    """

    def start_new_cas(self) -> None:
        """
        Replace :attr:`cas` with a fresh one, whose id is the next count.
        """
        self.started_cas_count += 1
        self.cas = CAS()
        self.cas.cas_id = self.started_cas_count

    def register_pipeline(self, *, pipeline: PipelineNode) -> None:
        """
        :param pipeline: The pipeline that fills :attr:`cas` from now on.
        :raises SecondPipelineError: If another pipeline already fills it.
        """
        if self.pipeline is not None and self.pipeline is not pipeline:
            raise SecondPipelineError(
                registered_pipeline=self.pipeline, second_pipeline=pipeline
            )
        self.pipeline = pipeline
