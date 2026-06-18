"""
Queries about a robot's past execution behaviour, expressed in the KRROOD Entity Query Language.

The plan mirrors the structure of the bullet-world demo
(coraplex/demos/coraplex_bullet_world_demo/demo.py): a PR2 parks its arms, raises its
torso, then transports three objects (milk, bowl, spoon) to a table.  Because the full
ROS / physics-simulator stack is not available in all environments, the actions are
represented by CodeNodes that carry a descriptive label — the plan graph structure,
timing, and status fields are identical to a live execution.

Each query is wrapped in a BehaviourQuery that pairs the natural-language question with
the EQL object so both can be inspected, logged, or evaluated together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import krrood.entity_query_language.factories as eql
from coraplex.datastructures.enums import TaskStatus
from coraplex.language import CodeNode
from coraplex.plans.factories import sequential, code
from coraplex.plans.plan_node import ActionNode, PlanNode


# ---------------------------------------------------------------------------
# BehaviourQuery — bundles a natural-language question with its EQL object
# ---------------------------------------------------------------------------

@dataclass
class BehaviourQuery:
    """A natural-language question paired with the EQL query that answers it."""

    question: str
    query: Any  # krrood SymbolicExpression / Query / Quantifier

    def evaluate(self):
        """Evaluate the EQL query and return results."""
        return self.query.evaluate()

    def __repr__(self) -> str:
        return f"BehaviourQuery({self.question!r})"


# ---------------------------------------------------------------------------
# Plan — mirrors the bullet-world demo structure
#
# sequential([
#     ParkArmsAction(Arms.BOTH),
#     MoveTorsoAction(TorsoState.HIGH),
#     TransportAction(milk,  target_pose, Arms.LEFT),
#     TransportAction(bowl,  target_pose, Arms.LEFT),
#     TransportAction(spoon, target_pose, Arms.LEFT, GraspDescription(...)),
# ])
#
# Each action is represented by a CodeNode (no-op lambda) so the plan can
# run without a live robot.  All real timing and status fields are set by
# plan.perform().
# ---------------------------------------------------------------------------

_ACTION_LABELS = [
    "ParkArmsAction(Arms.BOTH)",
    "MoveTorsoAction(TorsoState.HIGH)",
    "TransportAction(milk,   Pose(4.9, 3.3, 0.8),  Arms.LEFT)",
    "TransportAction(bowl,   Pose(5.0, 3.3, 0.75), Arms.LEFT)",
    "TransportAction(spoon,  Pose(5.1, 3.3, 0.75), Arms.LEFT, GraspDescription(FRONT, TOP))",
]

root = sequential([code(lambda: None) for _ in _ACTION_LABELS])

# Attach descriptive labels to leaf nodes in order
for node, label in zip(root.children, _ACTION_LABELS):
    node.label = label

plan = root.plan
root.perform()


# ---------------------------------------------------------------------------
# Queries — each uses a fresh variable so there is no cross-query aliasing
# ---------------------------------------------------------------------------

def _q_what_did_you_do() -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="What did you just do?",
        query=eql.an(
            eql.entity(n).where(n.is_leaf, n.status == TaskStatus.SUCCEEDED)
        ).ordered_by(n.start_time),
    )


def _q_walk_through_in_order() -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="Walk me through what you did in order.",
        query=eql.an(
            eql.entity(n).where(n.status == TaskStatus.SUCCEEDED)
        ).ordered_by(n.start_time),
    )


def _q_total_duration() -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="How long did the whole task take?",
        query=eql.the(eql.entity(n).where(n.parent == None)),  # noqa: E711
    )


def _q_duration_per_step() -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="How long did each step take?",
        query=eql.an(
            eql.entity(n).where(n.end_time != None)  # noqa: E711
        ).ordered_by(n.start_time),
    )


def _q_did_anything_go_wrong() -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="Did anything go wrong?",
        query=eql.an(eql.entity(n).where(n.status == TaskStatus.FAILED)),
    )


def _q_why_did_you_fail() -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="Why did you fail at that step?",
        query=eql.an(eql.entity(n.reason).where(n.status == TaskStatus.FAILED)),
    )


def _q_how_many_retries() -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="How many times did you retry before giving up?",
        query=(
            eql.set_of(c := eql.count_all())
            .where(n.status == TaskStatus.FAILED)
        ),
    )


def _q_which_fallback() -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="Which fallback did you end up using?",
        query=eql.an(
            eql.entity(n).where(
                n.status == TaskStatus.SUCCEEDED,
                eql.exists(
                    eql.variable(PlanNode, domain=n.left_siblings),
                    lambda s: s.status == TaskStatus.FAILED,
                ),
            )
        ),
    )


def _q_were_you_interrupted() -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="Were you ever interrupted? What caused it?",
        query=eql.an(eql.entity(n).where(n.status == TaskStatus.INTERRUPTED)),
    )


def _q_were_you_paused() -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="Was there a point where you were paused?",
        query=eql.an(eql.entity(n).where(n.status == TaskStatus.PAUSE)),
    )


def _q_longest_step() -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="Which step took the longest?",
        query=eql.max(
            n,
            key=lambda node: (node.end_time - node.start_time).total_seconds()
            if node.end_time is not None else 0.0,
        ),
    )


def _q_status_breakdown() -> BehaviourQuery:
    n = eql.variable(PlanNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="Were all subtasks successful, or did some fail?",
        query=(
            eql.set_of(status := n.status, c := eql.count(n))
            .grouped_by(status)
            .ordered_by(c, descending=True)
        ),
    )


def _q_world_modifications() -> BehaviourQuery:
    n = eql.variable(ActionNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="What world modifications did you make?",
        query=eql.an(
            eql.entity(n.execution_data.added_world_modifications)
            .where(n.status == TaskStatus.SUCCEEDED, n.execution_data != None)  # noqa: E711
        ),
    )


def _q_world_state_at_start() -> BehaviourQuery:
    n = eql.variable(ActionNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="What was the state of the world when you started the task?",
        query=eql.an(
            eql.entity(n.execution_data.execution_start_world_state)
            .where(n.parent == None, n.execution_data != None)  # noqa: E711
        ),
    )


def _q_world_state_at_end() -> BehaviourQuery:
    n = eql.variable(ActionNode, domain=plan.plan_graph.nodes())
    return BehaviourQuery(
        question="What was the state of the world when you finished?",
        query=eql.an(
            eql.entity(n.execution_data.execution_end_world_state)
            .where(n.parent == None, n.execution_data != None)  # noqa: E711
        ),
    )


queries: list[BehaviourQuery] = [
    _q_what_did_you_do(),
    _q_walk_through_in_order(),
    _q_total_duration(),
    _q_duration_per_step(),
    _q_did_anything_go_wrong(),
    _q_why_did_you_fail(),
    _q_how_many_retries(),
    _q_which_fallback(),
    _q_were_you_interrupted(),
    _q_were_you_paused(),
    _q_longest_step(),
    _q_status_breakdown(),
    _q_world_modifications(),
    _q_world_state_at_start(),
    _q_world_state_at_end(),
]


# ---------------------------------------------------------------------------
# Demo: print each question and evaluate
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for bq in queries:
        print(f"\n{'=' * 60}")
        print(f"  Q: {bq.question}")
        print(f"  {'─' * 56}")
        try:
            result = bq.evaluate()
            if hasattr(result, "__iter__"):
                items = list(result)
                if items:
                    for item in items:
                        label = getattr(item, "label", None) or repr(item)
                        print(f"    {label}")
                else:
                    print("    (no results)")
            else:
                label = getattr(result, "label", None) or repr(result)
                print(f"    {label}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
