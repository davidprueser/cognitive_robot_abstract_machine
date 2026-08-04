"""
Download Sage-10k meshes covering the object types that actually rest on room
floors, so room generation can dress a sampled piece with a mesh of its own
kind.

Without a broad cache, most sampled types have no mesh at all and
:meth:`~semantic_digital_twin.scene_generation.scene_schema._MeshTypeMatcher.random_match`
falls back to the whole pool -- which is why generated rooms ended up strewn
with whichever few types earlier demos happened to download.
"""

from __future__ import annotations

import argparse
import os

from sqlalchemy.orm import Session

from experiments.orm.ormatic_interface import *  # type: ignore  # noqa: F401,F403  registers ORM mappers
from experiments.scene_generation_experiments.data_preprocessing import (
    Sage10kSceneDownloader,
)
from experiments.scene_generation_experiments.utils import (
    DEFAULT_MINIMUM_CANDIDATES_PER_TYPE,
    download_meshes_for_floor_object_types,
)
from krrood.ormatic.utils import create_engine


def main() -> None:
    """
    Download meshes for every floor object type into the local scene cache.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minimum-candidates",
        type=int,
        default=DEFAULT_MINIMUM_CANDIDATES_PER_TYPE,
        help="Distinct meshes to make available per object type.",
    )
    arguments = parser.parse_args()

    session = Session(create_engine(os.environ["SEMANTIC_DIGITAL_TWIN_DATABASE_URI"]))
    sage10k_session = Session(create_engine(os.environ["SAGE10k_DATABASE_URI"]))
    downloader = Sage10kSceneDownloader(session=sage10k_session)

    achieved = download_meshes_for_floor_object_types(
        session, downloader, arguments.minimum_candidates
    )
    print(f"Cached meshes for {len(achieved)} object types.")


if __name__ == "__main__":
    main()
