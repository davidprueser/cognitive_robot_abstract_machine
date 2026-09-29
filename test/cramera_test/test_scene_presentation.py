"""
Authored appearance survives live export, object catalogs and recordings.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from cramera.live.bridge import Bridge
from cramera.live.live_bundle import build_live_scene
from cramera.live.recording_bundle import write_recording_bundle
from cramera.scene_presentation import PresentationField, ScenePresentation

from .test_live_bundle import (
    attached_bridge,
    laboratory_world,
    scene_payload,
    use_scratch_scenes_directory,
)
from .test_recording_bundle import frame_with_milk

# %% source presentation


@pytest.fixture
def source_presentation() -> dict:
    """
    Authored settings from a scene with independent object opt-ins.
    """
    return json.loads(
        (Path(__file__).parent / "dataset" / "authored_presentation.json").read_text()
    )


@pytest.fixture
def presented_bridge(source_presentation: dict) -> Bridge:
    """
    A native bridge carrying the source scene's rendering preferences.
    """
    bridge = attached_bridge(with_robot=True)
    bridge.presentation = ScenePresentation.from_json(source_presentation)
    return bridge


class TestPresentationScope:
    def test_bridge_accepts_an_optional_presentation(self, source_presentation):
        presentation = ScenePresentation.from_json(source_presentation)
        bridge = Bridge(presentation=presentation)
        bridge.attach(laboratory_world())

        assert bridge.presentation is presentation

    def test_only_source_appearance_is_applied(self, source_presentation):
        scene = {
            PresentationField.MODELS: [
                {PresentationField.ROBOT: False},
                {PresentationField.ROBOT: True},
            ],
            PresentationField.OBJECTS: [
                {PresentationField.KEY: entry[PresentationField.KEY]}
                for entry in source_presentation[PresentationField.OBJECTS]
            ],
        }

        ScenePresentation.from_json(source_presentation).apply_to_scene(scene)

        assert (
            scene[PresentationField.RENDERING]
            == source_presentation[PresentationField.RENDERING]
        )
        assert (
            scene[PresentationField.CAMERA]
            == source_presentation[PresentationField.CAMERA]
        )
        assert scene[PresentationField.MODELS] == [
            {
                PresentationField.ROBOT: False,
                PresentationField.PRESERVE_MATERIALS: True,
            },
            {PresentationField.ROBOT: True},
        ]
        assert scene[PresentationField.OBJECTS] == source_presentation[
            PresentationField.OBJECTS
        ][:1] + [
            {
                PresentationField.KEY: source_presentation[PresentationField.OBJECTS][
                    1
                ][PresentationField.KEY]
            }
        ]
        assert "laboratory" not in scene

    def test_robot_opt_in_does_not_preserve_environment(self, source_presentation):
        source_presentation[PresentationField.MODELS] = [
            model
            for model in source_presentation[PresentationField.MODELS]
            if model[PresentationField.ROBOT]
        ]

        presentation = ScenePresentation.from_json(source_presentation)

        assert presentation.preserve_environment_materials is False

    def test_material_preservation_requires_an_exact_object_key(
        self, source_presentation
    ):
        presentation = ScenePresentation.from_json(source_presentation)
        object_key = source_presentation[PresentationField.OBJECTS][0][
            PresentationField.KEY
        ]
        payload = {PresentationField.KEY: f"other/{object_key}"}

        presentation.apply_to_object(payload)

        assert PresentationField.PRESERVE_MATERIALS not in payload

    def test_output_edits_do_not_change_source_or_future_bundles(
        self, source_presentation
    ):
        original = deepcopy(source_presentation)
        presentation = ScenePresentation.from_json(source_presentation)
        scene = {}
        presentation.apply_to_scene(scene)
        scene[PresentationField.RENDERING].clear()
        scene[PresentationField.CAMERA].clear()
        following_scene = {}
        presentation.apply_to_scene(following_scene)

        assert source_presentation == original
        assert (
            following_scene[PresentationField.RENDERING]
            == original[PresentationField.RENDERING]
        )
        assert (
            following_scene[PresentationField.CAMERA]
            == original[PresentationField.CAMERA]
        )


# %% generated native scenes


class TestLivePresentation:
    def test_live_bundle_keeps_authored_settings(
        self, source_presentation, presented_bridge, monkeypatch, tmp_path
    ):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)

        build_live_scene(presented_bridge)
        scene = scene_payload(scenes)

        assert (
            scene[PresentationField.RENDERING]
            == source_presentation[PresentationField.RENDERING]
        )
        assert (
            scene[PresentationField.CAMERA]
            == source_presentation[PresentationField.CAMERA]
        )
        assert [
            model.get(PresentationField.PRESERVE_MATERIALS)
            for model in scene[PresentationField.MODELS]
        ] == [True, None]
        assert "laboratory" not in scene

    def test_live_object_catalog_keeps_authored_materials(self, presented_bridge):
        [entry] = presented_bridge.object_catalog()

        assert entry[PresentationField.PRESERVE_MATERIALS] is True

    def test_no_presentation_retains_existing_defaults(self, monkeypatch, tmp_path):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
        bridge = attached_bridge(with_robot=True)

        build_live_scene(bridge)
        scene = scene_payload(scenes)

        assert PresentationField.RENDERING not in scene
        assert PresentationField.CAMERA not in scene
        assert all(
            PresentationField.PRESERVE_MATERIALS not in entry
            for entry in scene[PresentationField.MODELS] + bridge.object_catalog()
        )

    def test_changed_presentation_invalidates_existing_bundle(
        self, presented_bridge, source_presentation, monkeypatch, tmp_path
    ):
        scenes = use_scratch_scenes_directory(monkeypatch, tmp_path)
        build_live_scene(presented_bridge)
        first = scene_payload(scenes)
        source_presentation[PresentationField.RENDERING] = {}
        presented_bridge.presentation = ScenePresentation.from_json(source_presentation)

        build_live_scene(presented_bridge)
        following = scene_payload(scenes)

        assert following["bundleSignature"] != first["bundleSignature"]
        assert (
            following[PresentationField.RENDERING]
            == source_presentation[PresentationField.RENDERING]
        )


class TestRecordedPresentation:
    def test_recording_keeps_authored_settings_and_object_materials(
        self, presented_bridge, source_presentation, tmp_path
    ):
        scene = write_recording_bundle(
            presented_bridge,
            [frame_with_milk()],
            20.0,
            tmp_path / "recording",
            "recording",
        )

        assert (
            scene[PresentationField.RENDERING]
            == source_presentation[PresentationField.RENDERING]
        )
        assert (
            scene[PresentationField.CAMERA]
            == source_presentation[PresentationField.CAMERA]
        )
        assert [
            model.get(PresentationField.PRESERVE_MATERIALS)
            for model in scene[PresentationField.MODELS]
        ] == [True, None]
        assert scene[PresentationField.OBJECTS][0][PresentationField.PRESERVE_MATERIALS]
        assert "laboratory" not in scene


class TestPresentationSignature:
    @pytest.mark.parametrize(
        "changed_field",
        [
            PresentationField.RENDERING,
            PresentationField.CAMERA,
            PresentationField.MODELS,
            PresentationField.OBJECTS,
        ],
    )
    def test_rendering_camera_and_environment_changes_invalidate_signature(
        self, source_presentation, changed_field
    ):
        first = ScenePresentation.from_json(source_presentation)
        source_presentation.pop(changed_field)
        following = ScenePresentation.from_json(source_presentation)

        assert first.signature() != following.signature()

    def test_reordered_source_dictionary_keeps_signature(self, source_presentation):
        first = ScenePresentation.from_json(source_presentation)
        following = ScenePresentation.from_json(
            dict(reversed(source_presentation.items()))
        )

        assert first.signature() == following.signature()
