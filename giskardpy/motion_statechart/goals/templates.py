from __future__ import division

from dataclasses import dataclass, field
from typing import List

from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.graph_node import (
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
)
from typing_extensions import Optional


@dataclass(repr=False, eq=False)
class Sequence(Goal):
    """
    Takes a list of nodes and wires their start/end conditions such that they are executed in order.
    Its observation is the observation of the last node in the sequence.
    """

    nodes: List[MotionStatechartNode] = field(default_factory=list, init=True)

    def expand(self, context: BuildContext) -> None:
        last_node: Optional[MotionStatechartNode] = None
        for i, node in enumerate(self.nodes):
            self.add_node(node)
            if last_node is not None:
                node.start_condition = last_node.observation_variable
            node.end_condition = node.observation_variable
            last_node = node

    def build(self, context: BuildContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self.nodes[-1].observation_variable)
