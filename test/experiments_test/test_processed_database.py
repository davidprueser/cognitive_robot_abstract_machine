from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
from experiments.orm.ormatic_interface import Base
from experiments.scene_generation_experiments.processed_database import (
    load_shelf_layers,
    load_shelves,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from semantic_digital_twin.scene_generation.scene_schema import (
    EGObject2D,
    EGShelf,
    EGShelfLayer,
    ObjectType,
)
from semantic_digital_twin.spatial_types import Pose2D
from semantic_digital_twin.world_description.geometry import Scale


def _session_with_one_shelf() -> Session:
    """
    A processed database holding one shelf, one layer and one object, written the same
    way :func:`preprocess_sage10k_for_training` writes them.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = Session(engine)
    shelf = EGShelf(
        scale=Scale(x=1.0, y=1.0, z=2.0),
        layers=[
            EGShelfLayer(
                objects=[
                    EGObject2D(
                        object_type=ObjectType.BOOK,
                        scale=Scale(x=0.1, y=0.1, z=0.2),
                        pose=Pose2D(x=0.1, y=0.2, yaw=0.3),
                        source_id="book_src",
                    )
                ],
                theme_dominant_type=ObjectType.BOOK,
            )
        ],
        theme_dominant_type=ObjectType.BOOK,
    )
    session.add(to_dao(shelf))
    session.commit()
    session.expunge_all()
    return session


# %% load_shelves and load_shelf_layers -- reading a stored object's pose back


def test_load_shelves_reads_each_objects_stored_pose() -> None:
    """
    EGObject2D stores its position and orientation as one Pose2D field rather than
    separate position/orientation fields, and the eager-loading query has to name the
    field that actually exists on the generated DAO or every read of a real shelf fails.
    """
    session = _session_with_one_shelf()

    [shelf] = load_shelves(session)

    [layer] = shelf.layers
    [obj] = layer.objects
    assert float(obj.pose.x) == pytest.approx(0.1)
    assert float(obj.pose.y) == pytest.approx(0.2)
    assert float(obj.pose.yaw) == pytest.approx(0.3)


def test_load_shelf_layers_reads_each_objects_stored_pose() -> None:
    session = _session_with_one_shelf()

    [layer] = load_shelf_layers(session)

    [obj] = layer.objects
    assert float(obj.pose.x) == pytest.approx(0.1)
    assert float(obj.pose.y) == pytest.approx(0.2)
    assert float(obj.pose.yaw) == pytest.approx(0.3)
