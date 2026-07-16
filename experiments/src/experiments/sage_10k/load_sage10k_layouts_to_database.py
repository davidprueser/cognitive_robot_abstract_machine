from __future__ import annotations

import os
import time
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine, drop_database
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.orm.ormatic_interface import Base

LAYOUTS_DIRECTORY = Path.home() / "Downloads" / "sage-10k-layouts"
COMMIT_BATCH_SIZE = 500


def main():
    """
    Drop the Sage10k database and reload it from all JSON layouts in
    ~/Downloads/sage-10k-layouts.

    Each subdirectory in that folder is expected to contain exactly one
    ``layout_*.json`` file. All scenes are committed in batches to keep
    memory usage bounded across the 10k entries.
    """
    engine = create_engine(os.getenv("SAGE10k_DATABASE_URI"))
    drop_database(engine)
    Base.metadata.create_all(engine)
    session = sessionmaker(engine)()

    loader = Sage10kDatasetLoader()
    layout_directories = sorted(directory for directory in LAYOUTS_DIRECTORY.iterdir() if directory.is_dir())
    total = len(layout_directories)
    print(f"Found {total} layouts. Loading into database...")

    start = time.time()
    for index, layout_directory in enumerate(layout_directories):
        scene = loader._parse_json(layout_directory)
        dao = to_dao(scene)
        session.add(dao)
        if (index + 1) % COMMIT_BATCH_SIZE == 0:
            session.commit()
            session.expire_all()
            print(f"  committed {index + 1}/{total}")

    session.commit()
    print(f"Done. Loaded {total} scenes in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    main()
