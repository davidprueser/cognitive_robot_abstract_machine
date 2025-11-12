import json

import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import TrinaryCondition, EndMotion
from giskardpy.motion_statechart.monitors.monitors import TrueMonitor
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
    LifeCycleState,
    ObservationState,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    KinematicStructureEntityKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3, TransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    trinary_logic_and,
    trinary_logic_not,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body
import numpy as np


@pytest.fixture()
def mini_world():
    world = World()
    with world.modify_world():
        body = Body(name=PrefixedName("root"))
        body2 = Body(name=PrefixedName("tip"))
        connection = RevoluteConnection.create_with_dofs(
            world=world, parent=body, child=body2, axis=Vector3.Z()
        )
        world.add_connection(connection)
    return world


def test_TrueMonitor():
    node = TrueMonitor(name=PrefixedName("muh"))
    json_data = node.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    node_copy = TrueMonitor.from_json(new_json_data)
    assert node_copy.name == node.name


def test_trinary_transition():
    msc = MotionStatechart()
    node1 = TrueMonitor(name=PrefixedName("muh1"))
    node2 = TrueMonitor(name=PrefixedName("muh2"))
    node3 = TrueMonitor(name=PrefixedName("muh3"))
    node4 = TrueMonitor(name=PrefixedName("muh4"))
    msc.add_node(node1)
    msc.add_node(node2)
    msc.add_node(node3)
    msc.add_node(node4)

    node1.start_condition = trinary_logic_and(
        node2.observation_variable,
        trinary_logic_and(
            node3.observation_variable, trinary_logic_not(node4.observation_variable)
        ),
    )
    condition = node1._start_condition
    json_data = condition.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    condition_copy = TrinaryCondition.from_json(new_json_data, motion_statechart=msc)
    assert condition_copy == condition


def test_to_json_joint_position_list(mini_world):
    connection = mini_world.connections[0]
    node = JointPositionList(
        name=PrefixedName("muh"),
        goal_state=JointState({connection: 0.5}),
        threshold=0.5,
    )
    json_data = node.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    node_copy = JointPositionList.from_json(new_json_data, world=mini_world)
    assert node_copy.name == node.name
    assert node_copy.threshold == node.threshold
    assert node_copy.goal_state == node.goal_state


def test_start_condition(mini_world):
    msc = MotionStatechart()
    node1 = TrueMonitor(name=PrefixedName("muh"))
    msc.add_node(node1)
    node2 = TrueMonitor(name=PrefixedName("muh2"))
    msc.add_node(node2)
    node3 = TrueMonitor(name=PrefixedName("muh3"))
    msc.add_node(node3)
    end = TrueMonitor(name=PrefixedName("done"))
    msc.add_node(end)

    node1.end_condition = node1.observation_variable
    node2.start_condition = node1.observation_variable
    node2.pause_condition = node3.observation_variable
    end.start_condition = trinary_logic_and(
        node2.observation_variable, node3.observation_variable
    )

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    msc_copy = MotionStatechart.from_json(new_json_data, world=mini_world)

    kin_sim = Executor(motion_statechart=msc_copy, world=mini_world)
    kin_sim.compile()
    for index, node in enumerate(msc.nodes):
        assert node.name == msc_copy.nodes[index].name
    assert len(msc.edges) == len(msc_copy.edges)
    for index, edge in enumerate(msc.edges):
        assert edge == msc_copy.edges[index]


def test_executing_json_parsed_statechart():
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        tip = Body(name=PrefixedName("tip"))
        tip2 = Body(name=PrefixedName("tip2"))
        ul = DerivativeMap()
        ul.velocity = 1
        ll = DerivativeMap()
        ll.velocity = -1
        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "a"), lower_limits=ll, upper_limits=ul
        )
        world.add_degree_of_freedom(dof)
        root_C_tip = RevoluteConnection(
            parent=root, child=tip, axis=Vector3.Z(), dof_name=dof.name
        )
        world.add_connection(root_C_tip)

        dof = DegreeOfFreedom(
            name=PrefixedName("dof", "b"), lower_limits=ll, upper_limits=ul
        )
        world.add_degree_of_freedom(dof)
        root_C_tip2 = RevoluteConnection(
            parent=root, child=tip2, axis=Vector3.Z(), dof_name=dof.name
        )
        world.add_connection(root_C_tip2)

    msc = MotionStatechart()

    task1 = JointPositionList(
        name=PrefixedName("task1"), goal_state=JointState({root_C_tip: 0.5})
    )
    always_true = TrueMonitor(name=PrefixedName("muh"))
    msc.add_node(always_true)
    msc.add_node(task1)
    end = EndMotion(name=PrefixedName("done"))
    msc.add_node(end)

    task1.start_condition = always_true.observation_variable
    end.start_condition = trinary_logic_and(
        task1.observation_variable, always_true.observation_variable
    )

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    msc_copy = MotionStatechart.from_json(new_json_data, world=world)

    kin_sim = Executor(
        motion_statechart=msc_copy,
        world=world,
        controller_config=QPControllerConfig.create_default_with_50hz(),
    )
    kin_sim.compile()

    task1_copy = msc_copy.get_node_by_name(task1.name)
    end_copy = msc_copy.get_node_by_name(end.name)
    assert task1_copy.observation_state == ObservationStateValues.UNKNOWN
    assert end_copy.observation_state == ObservationStateValues.UNKNOWN
    assert task1_copy.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end_copy.life_cycle_state == LifeCycleValues.NOT_STARTED
    msc_copy.draw("muh.pdf")
    kin_sim.tick_until_end()
    msc_copy.draw("muh.pdf")
    assert task1_copy.observation_state == ObservationStateValues.TRUE
    assert end_copy.observation_state == ObservationStateValues.TRUE
    assert task1_copy.life_cycle_state == LifeCycleValues.RUNNING
    assert end_copy.life_cycle_state == LifeCycleValues.RUNNING

    life_cycle_json = msc_copy.life_cycle_state.to_json()
    json_str = json.dumps(life_cycle_json)
    life_cycle_json_copy = json.loads(json_str)
    life_cycle_copy = LifeCycleState.from_json(
        life_cycle_json_copy, motion_statechart=msc_copy
    )
    assert life_cycle_copy == msc_copy.life_cycle_state

    observation_json = msc_copy.observation_state.to_json()
    json_str = json.dumps(observation_json)
    observation_json_copy = json.loads(json_str)
    observation_copy = ObservationState.from_json(
        observation_json_copy, motion_statechart=msc_copy
    )
    assert observation_copy == msc_copy.observation_state


def test_cart_goal_simple(pr2_world: World):
    tip = pr2_world.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world.get_kinematic_structure_entity_by_name("odom_combined")
    tip_goal = TransformationMatrix.from_xyz_quaternion(pos_x=-0.2, reference_frame=tip)

    msc = MotionStatechart()
    cart_goal = CartesianPose(
        name=PrefixedName("cart_goal"),
        root_link=root,
        tip_link=tip,
        goal_pose=tip_goal,
    )
    msc.add_node(cart_goal)
    end = EndMotion(name=PrefixedName("end"))
    msc.add_node(end)
    end.start_condition = cart_goal.observation_variable

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    tracker = KinematicStructureEntityKwargsTracker.from_world(pr2_world)
    kwargs = tracker.create_kwargs()
    msc_copy = MotionStatechart.from_json(new_json_data, **kwargs)

    kin_sim = Executor(
        motion_statechart=msc_copy,
        world=pr2_world,
        controller_config=QPControllerConfig.create_default_with_50hz(),
    )

    kin_sim.compile()
    kin_sim.tick_until_end()

    fk = pr2_world.compute_forward_kinematics_np(root, tip)
    assert np.allclose(fk, tip_goal.to_np(), atol=cart_goal.threshold)
