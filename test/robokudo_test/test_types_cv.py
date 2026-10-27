from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from robokudo.types.cv import ImageRotation

# %% image rotation


@pytest.mark.parametrize(
    "image_rotation, quarter_turns_counterclockwise",
    [
        (ImageRotation.QUARTER_TURN_COUNTERCLOCKWISE, 1),
        (ImageRotation.QUARTER_TURN_CLOCKWISE, -1),
        (ImageRotation.HALF_TURN, 2),
    ],
)
def test_rotation_turns_the_image(
    image_rotation: ImageRotation, quarter_turns_counterclockwise: int
) -> None:
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape((6, 8, 3))

    assert np.array_equal(
        image_rotation.rotate(image), np.rot90(image, k=quarter_turns_counterclockwise)
    )


@pytest.mark.parametrize(
    "image_rotation, rotated_size, rotated_intrinsic_matrix",
    [
        (
            ImageRotation.QUARTER_TURN_COUNTERCLOCKWISE,
            (6, 8),
            [[2.0, 0.0, 4.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]],
        ),
        (
            ImageRotation.QUARTER_TURN_CLOCKWISE,
            (6, 8),
            [[2.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]],
        ),
        (
            ImageRotation.HALF_TURN,
            (8, 6),
            [[1.0, 0.0, 5.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
        ),
    ],
)
def test_rotation_turns_the_intrinsics_with_the_image(
    image_rotation: ImageRotation,
    rotated_size: tuple[int, int],
    rotated_intrinsic_matrix: list[list[float]],
) -> None:
    intrinsics = o3d.camera.PinholeCameraIntrinsic(8, 6, 1.0, 2.0, 3.0, 4.0)

    rotated_intrinsics = image_rotation.rotate_intrinsics(intrinsics)

    assert (rotated_intrinsics.width, rotated_intrinsics.height) == rotated_size
    assert np.array_equal(
        rotated_intrinsics.intrinsic_matrix, np.array(rotated_intrinsic_matrix)
    )
