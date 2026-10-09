from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from experiments.orm.ormatic_interface import Base
from experiments.shelf_generation_experiments.training.processed_database import (
    ProcessedShelfDatabase,
)
from experiments.shelf_generation_experiments.training.shelf_model import ShelfModel
from krrood.ormatic.utils import create_engine

from .shelf_dataset import factorized_shelf_model


# %% fitted model
@pytest.fixture(scope="module")
def shelf_model() -> ShelfModel:
    """
    A fully factorized shelf model over synthetic shelves.
    """
    return factorized_shelf_model()


# %% meshes
@pytest.fixture
def scenes_root(tmp_path: Path) -> Path:
    """
    An empty directory to lay out downloaded scenes in.
    """
    root = tmp_path / "scenes"
    root.mkdir()
    return root


# %% processed database
@pytest.fixture(scope="session")
def processed_database(tmp_path_factory) -> ProcessedShelfDatabase:
    """
    A processed database shared by every test, since creating its schema is slow; each
    test stores and reads only records the others do not touch.
    """
    database_path = tmp_path_factory.mktemp("processed") / "processed.db"
    engine = create_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(engine)
    return ProcessedShelfDatabase(session=Session(engine))
