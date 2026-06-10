"""
Regression tests for the findings of the semantic_digital_twin package review.

Every test in this module asserts the *correct* (expected) behavior, so each test
fails as long as the corresponding bug exists and turns green once it is fixed.

Naming convention:
    test_bug_NN_*    -> numbered bugs from the review
    test_design_NN_* -> design problems / consistency risks from the review

Findings that are not crisply testable (performance characteristics,
docstring-only issues, the ActiveConnection1DOF.dof deepcopy-per-access trap)
are intentionally not covered here.
"""

import inspect
import os
import subprocess
import sys
import typing
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional
from uuid import uuid4

import numpy as np
import pytest

from krrood.adapters.json_serializer import from_json
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.callbacks.callback import Callback, ModelChangeCallback
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.reasoning.robot_predicates import (
    is_gripper_holding_something,
)
from semantic_digital_twin.reasoning.predicates import occluding_bodies
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.robot_parts import Camera, EndEffector, KinematicChain
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.spatial_types import Quaternion, Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.testing import world_setup
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
    SemanticAnnotation,
    WorldEntityWithID,
)
from semantic_digital_twin.world_description.world_state import WorldState

# %% Helpers


def _make_box_body(name: str, scale: Scale = Scale(1.0, 1.0, 1.0)) -> Body:
    body = Body(name=PrefixedName(name, prefix="review"))
    collision = Box(
        scale=scale,
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
    )
    body.collision = ShapeCollection([collision], reference_frame=body)
    return body


def _make_two_body_world() -> tuple:
    """World with a root body and one child body attached via FixedConnection."""
    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    child = _make_box_body("child")
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(child)
        connection = FixedConnection(parent=root, child=child)
        world.add_connection(connection)
    return world, root, child, connection


@dataclass(eq=False)
class ReviewEndEffector(EndEffector):
    """Minimal concrete EndEffector for predicate tests."""

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ):
        raise NotImplementedError


@dataclass(eq=False)
class ReviewKinematicChain(KinematicChain):
    """Minimal concrete KinematicChain for chain tests."""

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ):
        raise NotImplementedError


@dataclass(eq=False)
class ReviewCamera(Camera):
    """Minimal concrete Camera for predicate tests."""

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ):
        raise NotImplementedError


@dataclass(eq=False)
class ReviewAnnotation(SemanticAnnotation):
    """Semantic annotation with a mutable entity list, for hash-stability tests."""

    parts: List[Body] = field(default_factory=list)


@dataclass(eq=False)
class ReviewModelChangeCallback(ModelChangeCallback):
    def on_model_change(self, **kwargs):
        pass


# %% Bugs


def test_bug_01_world_str_contains_class_name():
    """world.py:486 uses self.__class__.name (the dataclass field default, None)
    instead of the class name, so every world stringifies as 'None v...'."""
    world = World()
    assert "World" in str(world)


@pytest.mark.skip("Not sure if this is wanted")
def test_bug_02_connection_t_child_expression_survives_json_roundtrip():
    """world_entity.py:842-864: Connection.to_json does not serialize
    connection_T_child_expression and _from_json does not restore it, although the
    MJCF adapter (adapters/mjcf.py:421) produces non-identity values."""
    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    child = Body(name=PrefixedName("child", prefix="review"))
    connection_T_child = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.0, y=2.0, z=3.0, child_frame=child
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(child)
        connection = FixedConnection(
            parent=root,
            child=child,
            connection_T_child_expression=connection_T_child,
        )
        world.add_connection(connection)

    tracker_kwargs = WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs()
    restored = from_json(connection.to_json(), **tracker_kwargs)

    np.testing.assert_allclose(
        restored.connection_T_child_expression.to_np(),
        connection.connection_T_child_expression.to_np(),
    )


def test_bug_03_body_inertial_survives_world_deepcopy():
    """world_entity.py:511-517: Body.copy_for_world only copies name/id/visual/
    collision, so deepcopying a world (which replays modifications using
    copy_for_world) silently resets all inertial properties."""
    world = World()
    body = _make_box_body("heavy_body")
    body.inertial = Inertial(mass=5.0)
    with world.modify_world():
        world.add_kinematic_structure_entity(body)

    copied_world = deepcopy(world)
    copied_body = copied_world.get_body_by_name("heavy_body")

    assert copied_body.inertial is not None
    assert copied_body.inertial.mass == pytest.approx(5.0)


def test_bug_04_omnidrive_translation_dofs_get_translation_limits():
    """connections.py:759-774: OmniDrive.create_with_dofs assigns the *rotation*
    velocity limits to x_vel/y_vel; the computed translation limits are unused."""
    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    base = Body(name=PrefixedName("base", prefix="review"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(base)
        drive = OmniDrive.create_with_dofs(
            world=world,
            parent=root,
            child=base,
            translation_velocity_limits=0.6,
            rotation_velocity_limits=0.5,
        )
        world.add_connection(drive)

    assert drive.x_velocity.limits.upper.velocity == pytest.approx(0.6)
    assert drive.y_velocity.limits.upper.velocity == pytest.approx(0.6)
    assert drive.yaw.limits.upper.velocity == pytest.approx(0.5)


def test_bug_05_has_collision_respects_volume_threshold():
    """world_entity.py:487-497: Body.has_collision documents and accepts volume/
    surface thresholds but ignores them entirely."""
    tiny_body = _make_box_body("tiny", scale=Scale(0.001, 0.001, 0.001))
    # volume = 1e-9 m^3, far below the documented default threshold of 1.001e-6 m^3
    assert tiny_body.has_collision() is False


def test_bug_06_callback_from_json_is_a_classmethod():
    """callbacks/callback.py:57: Callback._from_json is missing @classmethod, so
    deserialization through the SubclassJSONSerializer machinery mis-binds args."""
    assert isinstance(inspect.getattr_static(Callback, "_from_json"), classmethod)


def test_bug_07_empty_gripper_is_not_holding_something():
    """robot_predicates.py:186-189: is_gripper_holding_something checks
    len(branch(tool_frame)) > 0, but the branch always contains the tool frame
    itself, so the predicate is unconditionally True."""
    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    palm = _make_box_body("palm")
    tool_frame = Body(name=PrefixedName("tool_frame", prefix="review"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(palm)
        world.add_kinematic_structure_entity(tool_frame)
        world.add_connection(FixedConnection(parent=root, child=palm))
        world.add_connection(FixedConnection(parent=palm, child=tool_frame))
        gripper = ReviewEndEffector(
            name=PrefixedName("gripper", prefix="review"),
            root=palm,
            tool_frame=tool_frame,
            front_facing_orientation=Quaternion(0, 0, 0, 1),
        )
        world.add_semantic_annotation(gripper)

    # nothing is attached below the tool frame -> the gripper holds nothing
    assert is_gripper_holding_something(gripper) is False


def test_bug_08_robot_velocity_limit_setup_does_not_touch_environment_joints():
    """robot_parts.py:697,724-728: tighten_dof_velocity_limits_* iterate over ALL
    1-DOF connections of the world, so creating a robot rescales the velocity
    limits of unrelated environment joints (drawers, doors, ...)."""
    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    robot_base = _make_box_body("robot_base")
    robot_link = _make_box_body("robot_link")
    drawer_body = _make_box_body("drawer_body")

    env_limits = DegreeOfFreedomLimits(
        lower=DerivativeMap(None, -10.0, None, None),
        upper=DerivativeMap(None, 10.0, None, None),
    )
    with world.modify_world():
        for b in [root, robot_base, robot_link, drawer_body]:
            world.add_kinematic_structure_entity(b)
        world.add_connection(FixedConnection(parent=root, child=robot_base))
        robot_joint = RevoluteConnection.create_with_dofs(
            world=world,
            parent=robot_base,
            child=robot_link,
            axis=Vector3.Z(reference_frame=robot_base),
        )
        world.add_connection(robot_joint)
        drawer_joint = PrismaticConnection.create_with_dofs(
            world=world,
            parent=root,
            child=drawer_body,
            axis=Vector3.X(reference_frame=root),
            dof_limits=env_limits,
        )
        world.add_connection(drawer_joint)

    MinimalRobot.from_branch_in_world(robot_base)

    # the environment joint does not belong to the robot and must keep its limits
    assert drawer_joint.raw_dof.limits.upper.velocity == pytest.approx(10.0)
    assert drawer_joint.raw_dof.limits.lower.velocity == pytest.approx(-10.0)


def test_bug_09_kinematic_chain_with_root_equal_tip_has_no_connections():
    """robot_parts.py:352-354: KinematicChain.connections returns the *parent*
    connection of the root when root == tip, although that connection lies outside
    the chain. A zero-length chain has zero connections."""
    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    link = _make_box_body("link")
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(link)
        joint = RevoluteConnection.create_with_dofs(
            world=world, parent=root, child=link, axis=Vector3.Z(reference_frame=root)
        )
        world.add_connection(joint)
        chain = ReviewKinematicChain(
            name=PrefixedName("chain", prefix="review"), root=link, tip=link
        )
        world.add_semantic_annotation(chain)

    assert chain.connections == []


def test_bug_10_world_state_equality_is_order_independent():
    """world_state.py:191-208: WorldState.__eq__ checks DoF ids as a *set* but
    compares the data arrays positionally, so two states with identical per-DoF
    values in different column order compare unequal."""
    world = World()  # only needed for the lock
    dof_a, dof_b = uuid4(), uuid4()

    state_1 = WorldState(_world=world)
    state_1._add_dof(dof_a)
    state_1._add_dof(dof_b)
    state_2 = WorldState(_world=world)
    state_2._add_dof(dof_b)
    state_2._add_dof(dof_a)

    for state in (state_1, state_2):
        state[dof_a].position = 1.0
        state[dof_b].position = 2.0

    assert state_1 == state_2


def test_bug_11_nothing_occludes_a_body_in_clear_line_of_sight():
    """predicates.py:169-178: occluding_bodies mixes value-space and pixel-space
    indexing (seg[seg == idx].nonzero(), then tuple != int), which makes it return
    every visible body. With a clear line of sight it must return []."""
    world = World()
    root = Body(name=PrefixedName("root", prefix="review"))
    camera_body = Body(name=PrefixedName("camera_body", prefix="review"))
    target = _make_box_body("target")
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(camera_body)
        world.add_kinematic_structure_entity(target)
        world.add_connection(
            FixedConnection(
                parent=root,
                child=camera_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=1.0, reference_frame=root
                ),
            )
        )
        world.add_connection(
            FixedConnection(
                parent=root,
                child=target,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=3.0, z=1.0, reference_frame=root
                ),
            )
        )
        camera = ReviewCamera(
            name=PrefixedName("camera", prefix="review"),
            root=camera_body,
            forward_facing_axis=Vector3.X(),
            field_of_view=FieldOfView(horizontal_angle=0.99, vertical_angle=0.75),
        )
        world.add_semantic_annotation(camera)

    assert occluding_bodies(camera, target) == []


def test_bug_12_clearing_the_world_detaches_connections():
    """world.py:1944-1949: _clear_world_entities removes kinematic entities before
    connections; rustworkx drops the edges with the nodes, so the connection loop
    runs over an empty list and connection.remove_from_world() is never called."""
    world, root, child, connection = _make_two_body_world()

    with world.modify_world():
        world._clear_world_entities()

    assert connection._world is None


@pytest.mark.skip("Not sure if this is wanted")
def test_bug_13_world_reasoner_reason_works_as_documented():
    """world_reasoner.py:48-63: reason() calls add_semantic_annotation_recursively
    (an atomic modification requiring an open modify_world context) without opening
    one. The documented usage in doc/world_reasoner.md is a bare reason() call."""
    urdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "semantic_digital_twin",
        "resources",
        "urdf",
    )
    world = URDFParser.from_file(
        file_path=os.path.join(urdf_dir, "kitchen-small.urdf")
    ).parse()
    reasoner = WorldReasoner(world)

    result = reasoner.reason()

    assert isinstance(result, dict)


def test_bug_14_mixins_set_annotation_refers_to_typing_set():
    """mixins.py:13 imports random_events.set.Set, which shadows typing.Set in all
    'visited: Set[int]' annotations of the module."""
    hints = typing.get_type_hints(
        HasRootKinematicStructureEntity._kinematic_structure_entities
    )
    assert hints["visited"] in (typing.Set[int], set[int])


def test_bug_15_get_semantic_annotation_by_id_raises_package_exception():
    """world.py:1255: get_semantic_annotation_by_id raises a bare IndexError for
    unknown ids instead of WorldEntityNotFoundError like its sibling getters."""
    world = World()
    with pytest.raises(WorldEntityNotFoundError):
        world.get_semantic_annotation_by_id(uuid4())


# %% Design problems and consistency risks


@pytest.mark.skip("The hash behaviour is wanted, it ripping the hash table is not")
def test_design_01_hash_table_lookup_survives_annotation_mutation():
    """world_entity.py:614-625 + world.py:839: SemanticAnnotation.__hash__ depends
    on mutable fields, but the annotation is stored in _world_entity_hash_table
    under the hash it had at insertion time. Mutating the annotation afterwards
    (e.g. add_handle) strands the table entry under a stale key."""
    world, root, child, _ = _make_two_body_world()
    extra = _make_box_body("extra")
    with world.modify_world():
        world.add_kinematic_structure_entity(extra)
        world.add_connection(FixedConnection(parent=root, child=extra))
        annotation = ReviewAnnotation(
            name=PrefixedName("annotation", prefix="review"), parts=[child]
        )
        world.add_semantic_annotation(annotation)

    annotation.parts.append(extra)

    assert world._world_entity_hash_table.get(hash(annotation)) is annotation


@pytest.mark.skip(
    "I dont see a usecase why we would want two of the same callbacks yet. if there is a case then we change it"
)
def test_design_02_two_callbacks_of_same_class_are_both_registered():
    """world_entity.py:278-293: WorldEntityWithClassBasedID gives all instances of
    a class the same id and hash, so a second callback instance silently overwrites
    the first one in _world_entity_hash_table."""
    world = World()
    callback_1 = ReviewModelChangeCallback(_world=world)
    callback_2 = ReviewModelChangeCallback(_world=world)

    registered = list(world._world_entity_hash_table.values())
    assert any(entry is callback_1 for entry in registered)
    assert any(entry is callback_2 for entry in registered)


def test_design_03_modification_history_stays_consistent_after_exception():
    """world.py:205-219: when an exception escapes a modify_world block, the
    current modification block is discarded but the already-applied modifications
    are not rolled back. Replay-based operations (deepcopy, sync) then produce a
    different world than the original."""
    world = World()
    body_1 = Body(name=PrefixedName("body_1", prefix="review"))
    body_2 = Body(name=PrefixedName("body_2", prefix="review"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body_1)

    with pytest.raises(RuntimeError):
        with world.modify_world():
            world.add_kinematic_structure_entity(body_2)
            world.add_connection(FixedConnection(parent=body_1, child=body_2))
            raise RuntimeError("simulated user error")

    copied_world = deepcopy(world)

    # compare the raw graphs: the memoized .bodies property is itself stale after
    # the exception (see test_design_10)
    original_names = {b.name.name for b in world.kinematic_structure.nodes()}
    copied_names = {b.name.name for b in copied_world.kinematic_structure.nodes()}
    assert copied_names == original_names


def test_design_04_validation_still_works_with_python_optimize_flag():
    """world.py:488-508: World.validate consists of bare asserts, which are
    stripped under `python -O`, silently disabling all world validation."""
    snippet = (
        "from semantic_digital_twin.world import World\n"
        "from semantic_digital_twin.world_description.world_entity import Body\n"
        "from semantic_digital_twin.datastructures.prefixed_name import PrefixedName\n"
        "world = World()\n"
        "# build an invalid world (two disconnected roots) behind the back of the\n"
        "# modification machinery, then validate it: validation must fail\n"
        "world.kinematic_structure.add_node(Body(name=PrefixedName('a')))\n"
        "world.kinematic_structure.add_node(Body(name=PrefixedName('b')))\n"
        "world.validate()\n"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", snippet],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode != 0, (
        "an invalid world (two roots) passed validation under python -O:\n"
        + result.stdout
        + result.stderr
    )


def test_design_05_world_entity_equality_works_across_subclasses():
    """world_entity.py:113-116: WorldEntity.__eq__ returns False instead of
    NotImplemented when the type check fails. Because Python gives the subclass's
    reflected __eq__ priority, base_instance == subclass_instance evaluates the
    subclass's strict type check first and is False even for entities with the
    same id and hash."""

    @dataclass(eq=False)
    class EntityA(WorldEntityWithID):
        pass

    @dataclass(eq=False)
    class EntityB(EntityA):
        pass

    shared_id = uuid4()
    entity_a = EntityA(id=shared_id)
    entity_b = EntityB(id=shared_id)

    assert hash(entity_a) == hash(entity_b)
    assert entity_a == entity_b


def test_design_06_reset_state_context_restores_state_on_exception(world_setup):
    """world.py:147-157: ResetStateContextManager only restores the state when no
    exception occurred; a guard whose job is restoring state should restore it on
    the error path as well."""
    world, l1, l2, bf, r1, r2 = world_setup
    connection: PrismaticConnection = world.get_connection(l1, l2)
    connection.position = 0.0

    with pytest.raises(RuntimeError):
        with world.reset_state_context():
            connection.position = 1.0
            raise RuntimeError("simulated user error")

    assert connection.position == pytest.approx(0.0)


def test_design_07_merge_state_skips_dofs_missing_in_self():
    """world_state.py:381-391: merge_state documents 'overwriting values for any
    DOFs that are present in both states' but raises DofNotInWorldStateError for
    DoFs that only exist in the other state."""
    world = World()
    shared_dof, other_only_dof = uuid4(), uuid4()

    state_self = WorldState(_world=world)
    state_self._add_dof(shared_dof)
    state_other = WorldState(_world=world)
    state_other._add_dof(shared_dof)
    state_other._add_dof(other_only_dof)
    state_other[shared_dof].position = 4.2
    state_other[other_only_dof].position = 13.37

    state_self.merge_state(state_other)

    assert state_self[shared_dof].position == pytest.approx(4.2)
    assert other_only_dof not in state_self


def test_design_08_world_state_keys_does_not_expose_internal_list(world_setup):
    """world_state.py:210-212: keys() hands out the live _ids list; callers can
    corrupt the state bookkeeping by mutating the returned object."""
    world, *_ = world_setup
    state = world.state
    length_before = len(state)

    keys = state.keys()
    try:
        keys.append(uuid4())
    except AttributeError:
        pass  # a non-mutable view is a valid fix

    assert len(state) == length_before


def test_design_10_memoized_queries_match_graph_after_exception():
    """world.py:205-219 (found while reproducing test_design_03): the exception
    path of WorldModelUpdateContextManager.__exit__ never calls
    clear_memoization_cache and never bumps the model version, so memoized world
    queries (world.bodies etc.) keep returning pre-modification results that no
    longer match the actual kinematic structure."""
    world = World()
    body_1 = Body(name=PrefixedName("body_1", prefix="review"))
    body_2 = Body(name=PrefixedName("body_2", prefix="review"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body_1)

    with pytest.raises(RuntimeError):
        with world.modify_world():
            world.add_kinematic_structure_entity(body_2)
            raise RuntimeError("simulated user error")

    graph_names = {b.name.name for b in world.kinematic_structure.nodes()}
    memoized_names = {b.name.name for b in world.bodies}
    assert memoized_names == graph_names


def test_design_09_failed_atomic_modification_is_not_recorded():
    """world.py:283-289: atomic_world_modification appends the modification to the
    current block *before* executing the function. If the function raises and the
    caller catches the error inside the modify_world block, a phantom modification
    stays in the history."""
    world, root, child, _ = _make_two_body_world()

    outside_parent = Body(name=PrefixedName("outside_parent", prefix="review"))
    outside_child = Body(name=PrefixedName("outside_child", prefix="review"))

    with world.modify_world():
        bad_connection = FixedConnection(parent=outside_parent, child=outside_child)
        try:
            # bodies were never added to the world -> their graph indices are None
            world._add_connection(bad_connection)
        except Exception:
            pass

        block = world._model_manager.current_model_modification_block
        assert len(block) == 0, (
            "a failed atomic modification was recorded in the history: "
            f"{block.modifications}"
        )
