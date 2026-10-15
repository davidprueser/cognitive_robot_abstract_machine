from __future__ import annotations
import time
from unittest.mock import MagicMock

import pytest

from robokudo.utils.decorators import timer_decorator


class DummyTimerClass:
    @timer_decorator
    def timer(self) -> None:
        time.sleep(0.1)


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
