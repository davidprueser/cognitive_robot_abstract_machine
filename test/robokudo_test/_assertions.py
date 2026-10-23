from __future__ import annotations
import numpy as np
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


def assert_quaternion_equal_up_to_sign(
    actual_quaternion: npt.ArrayLike, expected_quaternion: npt.ArrayLike
) -> None:
    actual = np.array(actual_quaternion)
    expected = np.array(expected_quaternion)
    assert np.allclose(actual, expected) or np.allclose(actual, -expected)
