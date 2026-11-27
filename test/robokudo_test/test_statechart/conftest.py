"""
Fixtures of the tests of RoboKudo's statechart nodes: a context holding the CAS and an
executor ticking statecharts in it.
"""

from __future__ import annotations

# %% Imports

import pytest
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from semantic_digital_twin.world import World

from robokudo.statechart.cas_access import CASAccess

# %% CAS access and executor


@pytest.fixture()
def cas_access() -> CASAccess:
    """
    :return: The extension holding the CAS of :func:`statechart_executor`'s context.
    """
    return CASAccess()


@pytest.fixture()
def statechart_executor(cas_access: CASAccess) -> StatechartExecutor:
    """
    :return: An executor ticking statecharts in a context on an empty world that holds
        :func:`cas_access`.
    """
    context = StatechartContext.with_extensions(world=World(), extensions=[cas_access])
    return StatechartExecutor(context=context)
