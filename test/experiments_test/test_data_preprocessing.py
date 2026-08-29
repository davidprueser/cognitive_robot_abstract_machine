from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from experiments.orm.ormatic_interface import (
    Base,
    Sage10kObjectDAO,
    Sage10kPhysicallyBasedRenderingDAO,
    Sage10kPositionDAO,
    Sage10kRoomDAO,
    Sage10kRoomDAO_objects_association,
    Sage10kRotationDAO,
    Sage10kSceneDAO,
    Sage10kSceneDAO_rooms_association,
    Sage10kSizeDAO,
)
from experiments.scene_generation_experiments.data_preprocessing import (
    Sage10kSceneDownloader,
    SourceIdNotFoundError,
)
from krrood.ormatic.utils import create_engine


KNOWN_SOURCE_ID = "test_source_001"
KNOWN_ROOM_ID = "room_001"
KNOWN_SCENE_ID = "abc12345"


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    database_session = Session(engine)
    yield database_session
    database_session.close()


@pytest.fixture
def populated_session(session: Session) -> Session:
    pbr = Sage10kPhysicallyBasedRenderingDAO(metallic=0.0, roughness=0.5)
    object_position = Sage10kPositionDAO(x=1.0, y=2.0, z=0.0)
    object_rotation = Sage10kRotationDAO(x=0.0, y=0.0, z=90.0)
    object_dimensions = Sage10kSizeDAO(height=0.2, length=0.15, width=0.05)

    object_dao = Sage10kObjectDAO(
        id="obj_001",
        source_id=KNOWN_SOURCE_ID,
        room_id=KNOWN_ROOM_ID,
        type="book",
        description="A blue paperback novel on the shelf.",
        source="generation",
        place_id=KNOWN_ROOM_ID,
        place_guidance="on the bookshelf",
        mass=0.4,
        position=object_position,
        rotation=object_rotation,
        dimensions=object_dimensions,
        pbr_parameters=pbr,
    )

    room_position = Sage10kPositionDAO(x=0.0, y=0.0, z=0.0)
    room_dimensions = Sage10kSizeDAO(height=3.0, length=5.0, width=4.0)

    room_dao = Sage10kRoomDAO(
        id=KNOWN_ROOM_ID,
        room_type="living_room",
        floor_material="wood",
        position=room_position,
        dimensions=room_dimensions,
        objects=[Sage10kRoomDAO_objects_association(target=object_dao)],
    )

    scene_dao = Sage10kSceneDAO(
        id=KNOWN_SCENE_ID,
        building_style="modern",
        description="A modern apartment with one living room.",
        created_from_text="A modern apartment with one living room. Complete layout with doors/windows:",
        total_area=20.0,
        rooms=[Sage10kSceneDAO_rooms_association(target=room_dao)],
    )

    session.add(scene_dao)
    session.commit()
    return session


def test_raises_for_unknown_source_id(populated_session: Session) -> None:
    downloader = Sage10kSceneDownloader(session=populated_session)
    with pytest.raises(SourceIdNotFoundError):
        downloader.download_scene_for_source_id("nonexistent_source_id")


def test_downloads_scene_for_known_source_id(
    populated_session: Session, tmp_path: Path
) -> None:
    expected_scene_directory = tmp_path / f"20251230_layout_{KNOWN_SCENE_ID}"
    expected_scene_directory.mkdir()

    downloader = Sage10kSceneDownloader(session=populated_session)

    with patch.object(
        downloader.data_processor,
        "download_specific_scene",
        return_value=expected_scene_directory,
    ) as mock_download:
        result = downloader.download_scene_for_source_id(KNOWN_SOURCE_ID)

    mock_download.assert_called_once_with(KNOWN_SCENE_ID)
    assert result == expected_scene_directory