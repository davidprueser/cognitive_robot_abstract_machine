from __future__ import annotations

import json
import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from experiments.shelf_generation_experiments.preprocessing.raw_layout_import import (
    RawLayoutImport,
)
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.orm.ormatic_interface import Sage10kSceneDAO


# %% helpers
def _bundled_layouts() -> list[Path]:
    """
    :return: The layout files bundled with these tests.
    """
    return sorted((Path(__file__).parent / "layouts").glob("layout_*.json"))


def _downloaded_layouts(directory: Path) -> Path:
    """
    Lay the bundled layouts out the way they are downloaded: one directory per layout,
    each holding its ``layout_*.json`` file.

    :param directory: Where to lay the layouts out.
    :return: *directory*.
    """
    for layout_file in _bundled_layouts():
        layout_directory = directory / layout_file.stem
        layout_directory.mkdir(parents=True)
        shutil.copy(layout_file, layout_directory / layout_file.name)
    return directory


# %% import
def test_every_downloaded_layout_is_stored_as_a_scene(tmp_path: Path) -> None:
    database_uri = f"sqlite:///{tmp_path / 'raw.db'}"

    RawLayoutImport(
        layouts_directory=_downloaded_layouts(tmp_path / "layouts"),
        database_uri=database_uri,
    ).run()

    with Session(create_engine(database_uri)) as session:
        stored_ids = set(session.scalars(select(Sage10kSceneDAO.id)))
    assert stored_ids == {
        json.loads(layout_file.read_text())["id"] for layout_file in _bundled_layouts()
    }
