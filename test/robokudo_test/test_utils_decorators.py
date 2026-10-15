from __future__ import annotations
import time
from unittest.mock import MagicMock

import py_trees
import pytest

from robokudo.annotators.core import BaseAnnotator
from robokudo.utils.decorators import timer_decorator, record_time, publish_variables


class DummyTimerClass:
    def __init__(self) -> None:
        super().__init__()
        self.published = False

    @timer_decorator
    def timer(self) -> None:
        time.sleep(0.1)

    @publish_variables
    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS

    def publish_variables(self) -> None:
        self.published = True


class DummyAnnotator(BaseAnnotator):
    def __init__(self) -> None:
        super().__init__()
        self.published = False

    @record_time
    def compute(self, duration: float) -> None:
        time.sleep(duration)

    @publish_variables
    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS

    def publish_variables(self) -> None:
        self.published = True


class TestUtilsDecorators(object):
    def test_timer_decorator_standalone_method(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        @timer_decorator
        def timer() -> None:
            time.sleep(0.1)

        timer()
        captured = capsys.readouterr()
        assert "Function 'timer' took" in captured.out

    def test_timer_decorator_class_method_without_logger(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        dummy = DummyTimerClass()

        dummy.timer()
        captured = capsys.readouterr()
        assert "Function 'timer' took" in captured.out

    def test_timer_decorator_class_method_with_logger_self_kwargs(self) -> None:
        @timer_decorator
        def timer(self: DummyTimerClass) -> None:
            time.sleep(0.1)

        dummy = DummyTimerClass()
        dummy.rk_logger = MagicMock()

        timer(self=dummy)

        dummy.rk_logger.info.assert_called_once()
        expected_prefix = f"Function '{DummyTimerClass}.timer' took"
        assert expected_prefix in dummy.rk_logger.info.call_args[0][0]

    def test_timer_decorator_class_method_with_logger(self) -> None:
        dummy = DummyTimerClass()
        dummy.rk_logger = MagicMock()

        dummy.timer()
        dummy.rk_logger.info.assert_called_once()
        expected_prefix = f"Function '{DummyTimerClass}.timer' took"
        assert expected_prefix in dummy.rk_logger.info.call_args[0][0]

    def test_record_time(self) -> None:
        duration = 0.1
        dummy = DummyAnnotator()

        dummy.compute(duration)

        assert hasattr(dummy, "_times")
        assert dummy._times["compute"] == pytest.approx(
            duration, abs=0.01
        ), "Timing was not updated"

        dummy.compute(duration * 2)
        assert hasattr(dummy, "_times")
        assert dummy._times["compute"] == pytest.approx(
            duration * 2, abs=0.01
        ), "Timing was not updated"

    def test_publish_variables(self) -> None:
        dummy = DummyAnnotator()

        assert dummy.published == False
        dummy.update()
        assert dummy.published == True

    def test_publish_variables_non_annotator(self) -> None:
        dummy = DummyTimerClass()

        assert dummy.published == False
        assert pytest.raises(AssertionError, dummy.update)
        assert dummy.published == False
