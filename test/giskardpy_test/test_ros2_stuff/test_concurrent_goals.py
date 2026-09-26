import json
import threading
from dataclasses import dataclass, field
from typing_extensions import Any, List

import pytest
from action_msgs.msg import GoalStatus

from giskardpy.middleware.ros2.exceptions import WorldModelModifiedDuringMotionError
from giskardpy.middleware.ros2.ros2_interface import MyActionClient
from krrood.adapters.json_serializer import to_json

# %% mimics


@dataclass
class GoalFutureMimic:
    """
    Stands in for the future of a goal that the server has already accepted.
    """

    def done(self) -> bool:
        return True


@dataclass
class PayloadMimic:
    """
    Stands in for the result of the action, which carries the states as json.
    """

    result: str
    """
    The json the server wrote into the result.
    """


@dataclass
class ResultMimic:
    """
    Stands in for the result message the action client hands to a waiting caller.
    """

    status: int
    """
    The status the server finished the goal with.
    """

    result: PayloadMimic
    """
    The payload of the result.
    """


def succeeded_result() -> ResultMimic:
    """
    Build the result of a goal the server finished normally.
    """
    return ResultMimic(
        status=GoalStatus.STATUS_SUCCEEDED, result=PayloadMimic(result=json.dumps({}))
    )


def aborted_result(error: Exception) -> ResultMimic:
    """
    Build the result of a goal the server aborted with the given error.
    """
    return ResultMimic(
        status=GoalStatus.STATUS_ABORTED,
        result=PayloadMimic(result=json.dumps({"error": to_json(error)})),
    )


@dataclass
class StubbedActionClient(MyActionClient):
    """
    A :class:`MyActionClient` whose server is a thread instead of ros.

    Sending a goal starts a thread that waits for the given barrier and then reports the
    given result, which lets a test line up the waits of several clients without a
    running action server.
    """

    server_result: ResultMimic
    """
    What the stand-in server reports for every goal.
    """

    barrier: threading.Barrier | None = None
    """
    Released once every client of the test has sent its goal; None for a client that is
    the only one in its test.
    """

    _server_threads: List[threading.Thread] = field(default_factory=list)
    """
    The threads answering the goals sent so far.
    """

    def __post_init__(self):
        self._goal_counter = -1
        self._current_goal_id = None
        self.result = None
        self.action_name = "stubbed"

    def send_goal_async(self, goal) -> GoalFutureMimic:
        self._goal_counter += 1
        self._current_goal_id = self._goal_counter
        thread = threading.Thread(target=self._answer_goal, daemon=True)
        self._server_threads.append(thread)
        thread.start()
        return GoalFutureMimic()

    def _answer_goal(self) -> None:
        """
        Report the result once every client of the test is waiting for one.

        A broken barrier means another client never got as far as sending its goal; the
        result is reported anyway, so that the test fails on its assertion instead of
        hanging.
        """
        if self.barrier is not None:
            try:
                self.barrier.wait(timeout=10)
            except threading.BrokenBarrierError:
                pass
        self.result = self.server_result


def send_goal_in_thread(
    client: MyActionClient, outcomes: dict, key: str
) -> threading.Thread:
    """
    Start a thread that sends a goal and records what came back under the given key.

    :param client: client that sends the goal
    :param outcomes: dict collecting the result or the raised exception per thread
    :param key: name of the thread in that dict
    :return: the started thread
    """

    def target():
        try:
            outcomes[key] = client.send_goal(goal=None)
        except BaseException as e:
            outcomes[key] = e

    thread = threading.Thread(target=target, daemon=True, name=key)
    thread.start()
    return thread


# %% one client per thread


def test_two_threads_can_wait_for_their_goals_at_the_same_time():
    """
    Two clients, one per giskard, must be able to wait side by side; a loop shared by
    the process would let only the first thread wait and fail the second with "This
    event loop is already running".
    """
    barrier = threading.Barrier(2)
    clients = [
        StubbedActionClient(server_result=succeeded_result(), barrier=barrier)
        for _ in range(2)
    ]
    outcomes: dict[str, Any] = {}

    threads = [
        send_goal_in_thread(client, outcomes, f"goal {i}")
        for i, client in enumerate(clients)
    ]
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads)
    assert [type(outcome) for outcome in outcomes.values()] == [
        ResultMimic,
        ResultMimic,
    ]


# %% a single caller keeps its behaviour


def test_a_single_caller_gets_the_result_of_its_goal():
    client = StubbedActionClient(server_result=succeeded_result())

    result = client.send_goal(goal=None)

    assert result.status == GoalStatus.STATUS_SUCCEEDED
    assert client.result is None


def test_a_caller_of_an_aborted_goal_gets_the_reported_error():
    client = StubbedActionClient(
        server_result=aborted_result(WorldModelModifiedDuringMotionError())
    )

    with pytest.raises(WorldModelModifiedDuringMotionError):
        client.send_goal(goal=None)


def test_a_second_goal_can_be_sent_after_the_first_one_finished():
    """
    Every goal waits on an event loop of its own, so closing one may not stop the next
    one from waiting.
    """
    client = StubbedActionClient(server_result=succeeded_result())

    client.send_goal(goal=None)
    result = client.send_goal(goal=None)

    assert result.status == GoalStatus.STATUS_SUCCEEDED
