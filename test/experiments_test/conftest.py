from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from krrood.ormatic.utils import create_engine

from experiments.orm.ormatic_interface import *  # type: ignore


@pytest.fixture
def session() -> Session:
    """
    A session on a throwaway in-memory database holding the full experiments schema.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    database_session = Session(engine)
    yield database_session
    database_session.close()
