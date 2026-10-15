"""
Sensor data storage to filesystem.

This module provides an annotator for:

* Storing sensor data to local filesystem
* Supporting color and depth images
* Saving camera calibration info
* Using consistent naming schemes

The module uses:

* OpenCV image writing
* JSON serialization
* Timestamp-based filenames
* Configurable storage paths

.. note::
   Designed to work with robokudo.io.file_reader_interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import json
from pathlib import Path

import cv2
from py_trees.common import Status

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CASViews
from robokudo.utils.type_conversion import ros_camera_info_to_dict


@dataclass(kw_only=True)
class FileWriter(BaseAnnotator):
    """
    Sensor data storage to filesystem.

    This module provides methods to store sensor data into the local filesystem.
    This way you can work on recorded data without having to use bag files.
    It is intended to be used with robokudo.io.file_reader_interface

    .. note::
       Target directory must exist and be writable.
    """

    target_directory: Path = Path("/tmp/")
    """
    Existing directory the files are written to.
    """

    filename_prefix: str = "rk_"
    """
    Prefix of the file names; must match the prefix the file reader expects.
    """

    def generate_full_file_path_(self, cas_view: str, file_extension: str) -> str:
        """
        Generate the full filename and path where data should be stored.

        Creates a filepath using:

        * Configured prefix
        * Current timestamp
        * CAS view name
        * File extension

        :param cas_view: One of the definitions from CASView.X
        :param file_extension: File extension without dot (e.g. "jpg")
        :return: A string with the full path according to our naming scheme.
        """
        timestamp = self.get_cas().timestamp
        return str(
            self.target_directory
            / f"{self.filename_prefix}{timestamp}_{cas_view}.{file_extension}"
        )

    def compute(self) -> Status:
        """
        Process and store sensor data.

        The method:

        * Checks initialization status
        * Loads color image, depth image and camera info
        * Writes color image as JPG
        * Writes depth image as PNG
        * Saves camera info as JSON

        :return: SUCCESS after storing, FAILURE if not initialized
        """
        if not self.target_directory.is_dir():
            self.feedback_message = f"{self.target_directory} is not a directory"
            return Status.FAILURE

        color = self.get_cas().get(CASViews.COLOR_IMAGE)
        depth = self.get_cas().get(CASViews.DEPTH_IMAGE)
        camera_info = self.get_cas().get(CASViews.CAMERA_INFO)

        cv2.imwrite(
            self.generate_full_file_path_(
                cas_view=CASViews.COLOR_IMAGE, file_extension="jpg"
            ),
            color,
        )
        cv2.imwrite(
            self.generate_full_file_path_(
                cas_view=CASViews.DEPTH_IMAGE, file_extension="png"
            ),
            depth,
        )

        camera_info_dict = ros_camera_info_to_dict(camera_info)
        with open(
            str(
                self.generate_full_file_path_(
                    cas_view=CASViews.CAMERA_INFO, file_extension="json"
                )
            ),
            "w",
        ) as fp:
            json.dump(camera_info_dict, fp)

        return Status.SUCCESS
