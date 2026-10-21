import numpy as np
import pytest
from sensor_msgs.msg import Image

from robokudo.exceptions import (
    CVBridgeImageConversionError,
    CVBridgeImageShapeError,
    CVBridgeROSImagePayloadError,
    CVBridgeROSImageShapeError,
    CVBridgeROSImageStepError,
    CVBridgeUnsupportedImageData,
    CVBridgeUnsupportedEncoding,
    CVBridgeUnsupportedTargetEncoding,
)
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround


def _make_image_message(
    *,
    height: int,
    width: int,
    encoding: str,
    step: int,
    data: bytes,
    is_bigendian: int = 0,
) -> Image:
    message = Image()
    message.height = height
    message.width = width
    message.encoding = encoding
    message.is_bigendian = is_bigendian
    message.step = step
    message.data = data
    return message


class TestCVBridgeWorkaround(object):
    def test_imgmsg_to_cv2_rgb8_to_bgr8(self) -> None:
        bridge = CVBridgeWorkaround()
        red_green_blue = np.array(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [5, 6, 7]]],
            dtype=np.uint8,
        )

        message = _make_image_message(
            height=2,
            width=2,
            encoding="rgb8",
            step=2 * 3,
            data=red_green_blue.tobytes(),
        )
        blue_green_red = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

        assert blue_green_red.dtype == np.uint8
        assert blue_green_red.shape == (2, 2, 3)
        assert np.array_equal(blue_green_red, red_green_blue[:, :, ::-1])

    def test_imgmsg_to_cv2_handles_row_padding(self) -> None:
        bridge = CVBridgeWorkaround()
        # 3 pixels + 2 padding bytes per row
        raw = bytes([1, 2, 3, 255, 255, 4, 5, 6, 254, 254])
        message = _make_image_message(
            height=2,
            width=3,
            encoding="mono8",
            step=5,
            data=raw,
        )

        decoded = bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")

        assert decoded.dtype == np.uint8
        assert decoded.shape == (2, 3)
        assert np.array_equal(decoded, np.array([[1, 2, 3], [4, 5, 6]], np.uint8))

    def test_imgmsg_to_cv2_handles_bigendian_uint16(self) -> None:
        bridge = CVBridgeWorkaround()
        values = np.array([[0x0102, 0xA0B0]], dtype=np.uint16)
        message = _make_image_message(
            height=1,
            width=2,
            encoding="16UC1",
            step=2 * 2,
            data=values.astype(">u2").tobytes(),
            is_bigendian=1,
        )

        decoded = bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")

        assert decoded.dtype == np.uint16
        assert decoded.shape == (1, 2)
        assert np.array_equal(decoded, values)

    def test_imgmsg_to_cv2_rejects_invalid_ros_image_shape(self) -> None:
        bridge = CVBridgeWorkaround()
        message = _make_image_message(
            height=0,
            width=1,
            encoding="mono8",
            step=1,
            data=bytes([0]),
        )

        with pytest.raises(CVBridgeROSImageShapeError):
            bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")

    def test_imgmsg_to_cv2_rejects_small_ros_image_step(self) -> None:
        bridge = CVBridgeWorkaround()
        message = _make_image_message(
            height=1,
            width=2,
            encoding="rgb8",
            step=5,
            data=bytes([0, 0, 0, 0, 0]),
        )

        with pytest.raises(CVBridgeROSImageStepError):
            bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")

    def test_imgmsg_to_cv2_rejects_small_ros_image_payload(self) -> None:
        bridge = CVBridgeWorkaround()
        message = _make_image_message(
            height=2,
            width=2,
            encoding="mono8",
            step=2,
            data=bytes([0, 0, 0]),
        )

        with pytest.raises(CVBridgeROSImagePayloadError):
            bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")

    def test_imgmsg_to_cv2_casts_to_32fc1(self) -> None:
        bridge = CVBridgeWorkaround()
        depth_u16 = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        message = _make_image_message(
            height=2,
            width=2,
            encoding="16UC1",
            step=2 * 2,
            data=depth_u16.tobytes(),
        )

        decoded = bridge.imgmsg_to_cv2(message, desired_encoding="32FC1")

        assert decoded.dtype == np.float32
        assert decoded.shape == (2, 2)
        assert np.array_equal(decoded, depth_u16.astype(np.float32))

    def test_imgmsg_to_cv2_rejects_multi_channel_image_for_32fc1(self) -> None:
        bridge = CVBridgeWorkaround()
        blue_green_red = np.zeros((1, 1, 3), dtype=np.uint8)
        message = _make_image_message(
            height=1,
            width=1,
            encoding="bgr8",
            step=3,
            data=blue_green_red.tobytes(),
        )

        with pytest.raises(CVBridgeImageConversionError):
            bridge.imgmsg_to_cv2(message, desired_encoding="32FC1")

    def test_cv2_to_imgmsg_passthrough_roundtrip(self) -> None:
        bridge = CVBridgeWorkaround()
        image = np.array([[1, 2], [3, 4]], dtype=np.uint16)

        message = bridge.cv2_to_imgmsg(image, encoding="passthrough")
        restored = bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")

        assert message.encoding == "mono16"
        assert message.step == image.strides[0]
        assert np.array_equal(restored, image)

    def test_cv2_to_imgmsg_rejects_image_with_invalid_dimensions(self) -> None:
        bridge = CVBridgeWorkaround()
        image = np.zeros((1, 1, 1, 1), dtype=np.uint8)

        with pytest.raises(CVBridgeImageShapeError):
            bridge.cv2_to_imgmsg(image, encoding="passthrough")

    def test_cv2_to_imgmsg_rejects_encoding_mismatch(self) -> None:
        bridge = CVBridgeWorkaround()
        image = np.array([[1, 2], [3, 4]], dtype=np.uint16)

        with pytest.raises(ValueError, match="do not match requested encoding"):
            bridge.cv2_to_imgmsg(image, encoding="bgr8")

    def test_cv2_to_imgmsg_rejects_unsupported_image_data(self) -> None:
        bridge = CVBridgeWorkaround()
        image = np.array([[True, False]], dtype=np.bool_)

        with pytest.raises(CVBridgeUnsupportedImageData):
            bridge.cv2_to_imgmsg(image, encoding="passthrough")

    def test_imgmsg_to_cv2_rejects_unsupported_target_encoding(self) -> None:
        bridge = CVBridgeWorkaround()
        blue_green_red = np.zeros((1, 1, 3), dtype=np.uint8)
        message = _make_image_message(
            height=1,
            width=1,
            encoding="bgr8",
            step=3,
            data=blue_green_red.tobytes(),
        )

        with pytest.raises(CVBridgeUnsupportedTargetEncoding):
            bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")

    def test_imgmsg_to_cv2_rejects_unsupported_source_encoding(self) -> None:
        bridge = CVBridgeWorkaround()
        message = _make_image_message(
            height=1,
            width=1,
            encoding="not_an_encoding",
            step=1,
            data=bytes([0]),
        )

        with pytest.raises(CVBridgeUnsupportedEncoding):
            bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
