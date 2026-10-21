from __future__ import annotations
import queue

import pytest
from py_trees.blackboard import Blackboard

from robokudo.identifier import BlackboardIdentifier
from robokudo.utils.query import QueryHandler
from robokudo_msgs.action import Query


class TestUtilsQuery:
    def test_initialize_feedback_queue(self) -> None:
        blackboard = Blackboard()

        QueryHandler.initialize_feedback_queue()

        feedback_queue = blackboard.get(BlackboardIdentifier.QUERY_FEEDBACK)
        assert feedback_queue is not None
        assert isinstance(feedback_queue, queue.Queue)

        QueryHandler.initialize_feedback_queue()

        second_feedback_queue = blackboard.get(BlackboardIdentifier.QUERY_FEEDBACK)
        assert second_feedback_queue is not None
        assert isinstance(second_feedback_queue, queue.Queue)
        assert (
            second_feedback_queue == feedback_queue
        ), f"feedback queue should not be reinitialized"

    def test_get_feedback_queue(self) -> None:
        feedback_queue = QueryHandler.get_feedback_queue()
        assert feedback_queue is not None
        assert isinstance(feedback_queue, queue.Queue)

    def test_send_feedback(self) -> None:
        feedback = Query.Feedback()
        feedback.feedback = "Test Feedback"
        QueryHandler.send_feedback(feedback)

        feedback_queue = Blackboard().get(BlackboardIdentifier.QUERY_FEEDBACK)
        feedback_item = feedback_queue.get()
        assert feedback_item == feedback
        assert feedback_item.feedback == feedback.feedback

    def test_send_feedback_string(self) -> None:
        feedback = "Test Feedback"
        QueryHandler.send_feedback_string(feedback_string=feedback)

        feedback_queue = Blackboard().get(BlackboardIdentifier.QUERY_FEEDBACK)
        feedback_item = feedback_queue.get()
        assert feedback_item.feedback == feedback

    def test_send_answer(self) -> None:
        answer = Query.Result()
        answer.text_result = "Test Answer"
        QueryHandler.send_answer(answer)

        answer_item = Blackboard().get(BlackboardIdentifier.QUERY_ANSWER)
        assert answer_item == answer
        assert answer_item.text_result == answer.text_result

    def test_send_answer_invalid_type(self) -> None:
        answer = "Test Answer"
        assert pytest.raises(TypeError, QueryHandler.send_answer, answer)

    @pytest.mark.parametrize("answer", ["Test Answer", 42, 42.0])
    def test_send_arbitrary_answer(self, answer: str | int | float) -> None:
        QueryHandler.send_arbitrary_answer(answer)

        answer_item = Blackboard().get(BlackboardIdentifier.QUERY_ANSWER)
        assert answer_item == answer

    def test_preempt_requested(self) -> None:
        blackboard = Blackboard()

        # Works uninitialized
        is_requested = QueryHandler.preempt_requested()
        assert is_requested == False

        blackboard.set(BlackboardIdentifier.QUERY_PREEMPT_REQUESTED, True)

        # Works with preempt requested
        is_requested = QueryHandler.preempt_requested()
        assert is_requested == True

        blackboard.set(BlackboardIdentifier.QUERY_PREEMPT_REQUESTED, False)

        # Works with no preempt requested
        is_requested = QueryHandler.preempt_requested()
        assert is_requested == False

    def test_acknowledge_preempt_request(self) -> None:
        blackboard = Blackboard()

        assert pytest.raises(
            KeyError, blackboard.get, BlackboardIdentifier.QUERY_PREEMPT_ACKNOWLEDGED
        )

        QueryHandler.acknowledge_preempt_request()

        assert blackboard.get(BlackboardIdentifier.QUERY_PREEMPT_ACKNOWLEDGED) == True
