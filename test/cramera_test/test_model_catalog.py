"""
The model catalog derives choices from CRAM's semantic robot annotations.
"""

from pathlib import Path

import pytest

from cramera.model_catalog import (
    BuilderStep,
    CatalogField,
    EnvironmentKind,
    EnvironmentModel,
    ModelCatalog,
    RobotModel,
)
from semantic_digital_twin.adapters.package_resolver import CompositePathResolver
from semantic_digital_twin.predetermined_maps.apartment_environment import (
    ApartmentEnvironment,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import CheezeIt
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.minimal_robot import MinimalRobot

# %% registered models


def test_catalog_includes_supported_annotations(supported_abstract_robots) -> None:
    """
    Every robot supported by the shared CRAM fixture is offered.

    :param supported_abstract_robots: CRAM's existing supported annotation inventory.
    """
    assert set(supported_abstract_robots) <= {
        model.annotation for model in ModelCatalog.installed().robots
    }


@pytest.mark.parametrize(
    "annotation, excluded",
    [
        (Tracy, {BuilderStep.NAVIGATE, BuilderStep.MOVE_TORSO}),
        (MinimalRobot, set(BuilderStep)),
    ],
)
def test_capabilities_follow_robot_parts(annotation, excluded) -> None:
    """
    Unavailable operations are omitted from the annotated robot's palette.

    :param annotation: The robot annotation to inspect.
    :param excluded: Operations the annotation cannot support.
    """
    assert not (set(RobotModel(annotation).steps) & excluded)


def test_mobile_manipulator_has_every_builder_step() -> None:
    """
    PR2's nested base, torso and arms expose the complete palette.
    """
    assert set(RobotModel(PR2).steps) == set(BuilderStep)


def test_one_arm_robot_uses_coraplex_default_arm() -> None:
    """
    A single arm is selected through CRAM's BOTH/default-arm convention.
    """
    assert RobotModel(HSRB).arms == ["BOTH"]


def test_environment_catalog_matches_installed_worlds() -> None:
    """
    Every installed URDF environment is offered without another HTML list.
    """
    catalog = ModelCatalog.installed()
    assert {Path(model.path) for model in catalog.environments} == set(
        catalog.worlds_directory.glob("*.urdf")
    )


def test_payload_contains_imports_and_capabilities() -> None:
    """
    The frontend receives the concrete import and supported operation names.
    """
    payload = RobotModel(Tracy).to_payload()
    assert payload["import"] == f"from {Tracy.__module__} import {Tracy.__name__}"
    assert payload["steps"] == [step.value for step in RobotModel(Tracy).steps]
    assert payload["arms"] == ["LEFT", "RIGHT", "BOTH"]


def test_catalog_payload_matches_installed_models() -> None:
    """
    Every domain entry is represented in the browser's catalog.
    """
    catalog = ModelCatalog.installed()
    payload = catalog.to_payload()
    assert payload["ok"] is True
    assert payload["robots"] == [robot.to_payload() for robot in catalog.robots]
    assert payload["environments"] == [
        environment.to_payload() for environment in catalog.environments
    ]


# %% looking around


def test_a_robot_with_a_camera_may_look_at_a_point() -> None:
    assert BuilderStep.LOOK_AT in RobotModel(Tracy).steps


def test_a_robot_without_a_camera_may_not() -> None:
    assert BuilderStep.LOOK_AT not in RobotModel(MinimalRobot).steps


# %% robots other distributions register


def test_a_robot_another_distribution_registers_is_offered(monkeypatch) -> None:
    from importlib.metadata import EntryPoint

    from cramera import model_catalog

    from .dataset.standing_robot import StandingRobot

    registered = EntryPoint(
        name="standing_robot",
        value=f"{StandingRobot.__module__}:{StandingRobot.__name__}",
        group=model_catalog.ROBOT_ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(
        model_catalog,
        "entry_points",
        lambda group: [registered] if group == registered.group else [],
    )

    assert StandingRobot in {
        model.annotation for model in ModelCatalog.installed().robots
    }


# %% the real lab, built by a map class rather than read from a file


def test_the_apartment_map_is_offered_beside_the_world_files() -> None:
    """
    The real lab is not a file coraplex ships but a map class that populates a world, so
    the catalog lists it separately from the files.
    """
    assert [model.map for model in ModelCatalog.installed().maps] == [
        ApartmentEnvironment
    ]


def test_a_map_environment_tells_the_browser_how_to_import_it() -> None:
    [model] = ModelCatalog.installed().maps
    payload = model.to_payload()

    assert payload[CatalogField.KIND] == EnvironmentKind.MAP
    assert payload[CatalogField.CLASS] == ApartmentEnvironment.__name__
    assert (
        payload[CatalogField.IMPORT]
        == f"from {ApartmentEnvironment.__module__} import {ApartmentEnvironment.__name__}"
    )
    assert payload[CatalogField.NAME]


def test_a_world_file_is_marked_as_a_file() -> None:
    payload = EnvironmentModel("/worlds/apartment.urdf").to_payload()

    assert payload[CatalogField.KIND] == EnvironmentKind.FILE
    assert payload[CatalogField.PATH] == "/worlds/apartment.urdf"


def test_the_catalog_payload_lists_the_maps() -> None:
    catalog = ModelCatalog.installed()

    assert catalog.to_payload()[CatalogField.MAPS] == [
        model.to_payload() for model in catalog.maps
    ]


def test_no_map_is_named_like_a_world_file() -> None:
    """
    The browser lists files and maps in one choice, so their names must not collide.
    """
    catalog = ModelCatalog.installed()
    names = [model.to_payload()[CatalogField.NAME] for model in catalog.environments]
    names += [model.to_payload()[CatalogField.NAME] for model in catalog.maps]

    assert len(set(names)) == len(names), names


# %% looking for an object


def test_a_robot_with_a_camera_may_detect_an_object() -> None:
    assert BuilderStep.DETECT in RobotModel(Tracy).steps


def test_a_robot_without_a_camera_may_not_detect() -> None:
    assert BuilderStep.DETECT not in RobotModel(MinimalRobot).steps


# %% objects with an annotation class


def test_the_cheeze_it_is_offered_as_a_typed_object() -> None:
    """
    The only object the real-lab demonstration proves: detected by its annotation class
    and spawned as an annotation of it.
    """
    assert [model.annotation for model in ModelCatalog.installed().objects] == [
        CheezeIt
    ]


def test_a_typed_object_tells_the_browser_its_class_and_mesh() -> None:
    [model] = ModelCatalog.installed().objects
    payload = model.to_payload()

    assert payload[CatalogField.NAME] == CheezeIt.__name__
    assert payload[CatalogField.CLASS] == CheezeIt.__name__
    assert (
        payload[CatalogField.IMPORT]
        == f"from {CheezeIt.__module__} import {CheezeIt.__name__}"
    )
    assert payload[CatalogField.MESH] == model.mesh
    assert payload[CatalogField.MESH_URL] == model.mesh_url


def test_a_typed_objects_mesh_is_the_apartments(apartment_meshes) -> None:
    """
    The cereal's mesh ships with the apartment package, where the map resolves it.
    """
    [model] = ModelCatalog.installed().objects

    assert CompositePathResolver().resolve(
        model.mesh_url
    ) == ApartmentEnvironment.mesh_path(model.mesh)


def test_the_catalog_payload_lists_the_typed_objects() -> None:
    catalog = ModelCatalog.installed()

    assert catalog.to_payload()[CatalogField.OBJECTS] == [
        model.to_payload() for model in catalog.objects
    ]
