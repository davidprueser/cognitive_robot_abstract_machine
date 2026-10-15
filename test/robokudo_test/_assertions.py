from __future__ import annotations
import numpy as np
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


def assert_quat_equal_up_to_sign(
    actual_quat: npt.ArrayLike, expected_quat: npt.ArrayLike
) -> None:
    actual = np.array(actual_quat)
    expected = np.array(expected_quat)
    assert np.allclose(actual, expected) or np.allclose(actual, -expected)
