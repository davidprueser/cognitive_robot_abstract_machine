import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
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
    directory: Path = field(
        default_factory=lambda: Path.home() / "Documents" / "sage-10k-scenes"
    )

    def get_objects_from_scenes(self, scene_paths: Dict[Path, str]):
        objects = []
        for path, source_id in scene_paths:
            objects += self._get_objects_from_scene(
                path,
            )
        return objects

    def download_specific_scene(self, layout_name: str):
        """
        Download a specific scene from the Sage10k dataset by its layout name.

        :param layout_name: The name of the layout to download. Should be of form "*_layout_*id*"
        :return: The path to the downloaded scene.
        """
        from huggingface_hub import list_repo_files

        files = list_repo_files(repo_id="nvidia/SAGE-10k", repo_type="dataset")

        [matching_file] = [
            f
            for f in files
            if f.startswith("scenes/") and f.endswith(f"_{layout_name}.zip")
        ]
        print(matching_file)

        base_url = f"https://huggingface.co/datasets/nvidia/SAGE-10k/resolve/main/"
        return self._download_scene_if_not_exists(base_url + matching_file)

    def _download_scene_if_not_exists(self, scene_url: str) -> Path:
        """
        Download the scene from the Sage10k dataset and unzip it.
        Returns early if a directory with the requested scene already exists.

        :param scene_url: The URL of the scene to be downloaded.
        :return: The path to the unzipped scene.
        """
        self.directory.mkdir(parents=True, exist_ok=True)

        filename = Path(urlparse(scene_url).path).name
        zipped_scene = self.directory / filename
        extraction_directory = self.directory / zipped_scene.stem

        # return early if the scene exists already
        if extraction_directory.exists():
            return extraction_directory

        # download the scene
        with requests.get(scene_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with zipped_scene.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        # unzip the scene
        extraction_directory.mkdir(parents=True, exist_ok=True)
        print(zipped_scene)
        with ZipFile(zipped_scene, "r") as zip_ref:
            zip_ref.extractall(extraction_directory)

        os.remove(str(zipped_scene))
        logger.info(f"Downloaded and extracted {scene_url} to {extraction_directory}")
        return extraction_directory
