from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from experiments.shelf_generation_experiments.dataset_environment import (
    DatasetEnvironmentVariable,
)
from experiments.shelf_generation_experiments.preprocessing.record_writer import (
    BatchedRecordWriter,
)
from krrood.ormatic.utils import create_engine, drop_database
from semantic_digital_twin.adapters.sage_10k_dataset.loader import (
    Sage10kDatasetLoader,
)
from semantic_digital_twin.orm.ormatic_interface import Base


# %% raw layout import
@dataclass
class RawLayoutImport:
    """
    Imports downloaded sage10k layouts into the raw database that preprocessing reads.
    """

    layouts_directory: Path
    """
    Directory holding one sub-directory per layout, each with exactly one
    ``layout_*.json`` file.
    """

    database_uri: str
    """
    Connection string of the raw database.
    """

    commit_batch_size: int = 500
    """
    How many layouts to stage before committing them.
    """

    loader: Sage10kDatasetLoader = field(default_factory=Sage10kDatasetLoader)
    """
    Parses a layout directory into a scene.
    """

    def run(self) -> None:
        """
        Drop and recreate the raw database and store every layout of
        :attr:`layouts_directory` in it.
        """
        engine = create_engine(self.database_uri)
        drop_database(engine)
        Base.metadata.create_all(engine)
        layout_directories = sorted(
            directory
            for directory in self.layouts_directory.iterdir()
            if directory.is_dir()
        )
        BatchedRecordWriter(
            session=Session(engine),
            label="layouts",
            commit_batch_size=self.commit_batch_size,
        ).store_all(
            self.loader._parse_json(directory) for directory in layout_directories
        )


# %% command-line entry point
def main() -> None:
    """
    Import the layouts found under the directory the environment names into the raw
    database the environment names.
    """
    RawLayoutImport(
        layouts_directory=Path(DatasetEnvironmentVariable.RAW_LAYOUTS_ROOT.read()),
        database_uri=DatasetEnvironmentVariable.RAW_DATABASE_URI.read(),
    ).run()


if __name__ == "__main__":
    main()
