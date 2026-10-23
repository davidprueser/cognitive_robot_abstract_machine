"""
Tests for reporting the semantic annotations that join a world.
"""

from __future__ import annotations

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Noodles
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.examples.new_semantic_annotation_reporter import (
    NewSemanticAnnotationReporter,
)

# %% reporting


class TestReportingNewSemanticAnnotations:
    def test_annotation_that_joined_is_reported_once_with_its_class_label(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        world = World()
        with world.modify_world():
            root = Body(name=PrefixedName("map"))
            world.add_body(root)
        reporter = NewSemanticAnnotationReporter(world=world)
        new_body = Body(name=PrefixedName("new_body"))
        annotation = Noodles(root=new_body, class_label="noodles_box")
        with world.modify_world():
            world.add_connection(
                FixedConnection.create_with_dofs(
                    world=world, parent=root, child=new_body
                )
            )
            world.add_semantic_annotation(annotation)

        reporter.report()
        reporter.report()

        reported_lines = capsys.readouterr().out.splitlines()
        assert [
            (str(annotation.id) in line, annotation.class_label in line)
            for line in reported_lines
        ] == [(True, True)]
