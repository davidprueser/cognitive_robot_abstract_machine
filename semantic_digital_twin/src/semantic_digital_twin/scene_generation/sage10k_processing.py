from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile

import requests


logger = logging.getLogger(__name__)

try:
    import huggingface_hub
except ImportError:
    logger.warning(
        "huggingface_hub not installed. `Sage10kDatasetLoader.available_scenes` will not work."
        "Install it with `pip install huggingface_hub`."
    )
    huggingface_hub = None


@dataclass
class EGDataProcessing:
    """
    Handles downloading and local caching of Sage-10k scene assets.

    Scenes are stored under :attr:`directory`, one sub-folder per scene.
    """

    directory: Path = field(
        default_factory=lambda: Path.home() / "Documents" / "sage-10k-scenes"
    )
    """
    Root directory where downloaded scenes are stored.
    """

    def download_specific_scene(self, layout_name: str) -> Path:
        """
        Download a specific scene from the Sage10k dataset by its layout name.

        :param layout_name: The name of the layout to download. Should
            be of form ``*_layout_*id*``.
        :return: The path to the downloaded scene.
        """
        from huggingface_hub import list_repo_files

        files = list_repo_files(repo_id="nvidia/SAGE-10k", repo_type="dataset")

        [matching_file] = [
            f
            for f in files
            if f.startswith("scenes/") and f.endswith(f"_{layout_name}.zip")
        ]
        logger.info("Downloading scene file: %s", matching_file)

        base_url = "https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/"
        return self._download_scene_if_not_exists(base_url + matching_file)

    def _download_scene_if_not_exists(self, scene_url: str) -> Path:
        """
        Download the scene from the Sage10k dataset and unzip it.

        Returns early if a directory with the requested scene already
        exists.

        :param scene_url: The URL of the scene to be downloaded.
        :return: The path to the unzipped scene.
        """
        self.directory.mkdir(parents=True, exist_ok=True)

        filename = Path(urlparse(scene_url).path).name
        zipped_scene = self.directory / filename
        extraction_directory = self.directory / zipped_scene.stem

        if extraction_directory.exists():
            return extraction_directory

        with requests.get(scene_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with zipped_scene.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file_handle.write(chunk)

        extraction_directory.mkdir(parents=True, exist_ok=True)
        with ZipFile(zipped_scene, "r") as zip_file:
            zip_file.extractall(extraction_directory)

        zipped_scene.unlink()
        logger.info("Downloaded and extracted %s to %s", scene_url, extraction_directory)
        return extraction_directory
