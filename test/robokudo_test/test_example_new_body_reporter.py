"""
Tests for reporting the bodies that join a world.
"""

from __future__ import annotations

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.examples.new_body_reporter import NewBodyReporter

# %% reporting


class TestReportingNewBodies:
    def test_body_that_joined_is_reported_once(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        world = World()
        with world.modify_world():
            root = Body(name=PrefixedName("map"))
            world.add_body(root)
        reporter = NewBodyReporter(world=world)
        new_body = Body(name=PrefixedName("new_body"))
        with world.modify_world():
            world.add_connection(
                FixedConnection.create_with_dofs(
                    world=world, parent=root, child=new_body
                )
            )

        reporter.report()
        reporter.report()

        reported_lines = capsys.readouterr().out.splitlines()
        assert [str(new_body.id) in line for line in reported_lines] == [True]
