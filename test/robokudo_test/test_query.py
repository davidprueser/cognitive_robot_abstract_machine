from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
import json
import multiprocessing
import threading
from pathlib import Path

import py_trees
import pytest

from action_msgs.msg import GoalStatus
from py_trees.blackboard import Blackboard

from krrood.adapters.json_serializer import from_json

import robokudo.cas
import robokudo.defs
import robokudo.descriptors.camera_configs.config_filereader_playback
import robokudo.descriptors.camera_configs.config_mongodb_playback
import robokudo.garden
import robokudo.idioms
import robokudo.io.file_reader_interface
import robokudo.io.storage_reader_interface
import robokudo.types.annotation
import robokudo.types.scene
import robokudo.utils.tree_execution
import robokudo.utils.data_downloader
from robokudo.annotators.collection_reader import CollectionReaderAnnotator
from robokudo.annotators.image_preprocessor import ImagePreprocessorAnnotator
from robokudo.annotators.plane import PlaneAnnotator
from robokudo.annotators.pointcloud_cluster_extractor import PointCloudClusterExtractor
from robokudo.annotators.pointcloud_crop import PointcloudCropAnnotator
from robokudo.annotators.query import QueryAnnotator, GenerateQueryResult
from robokudo.belief_state import BeliefState
from robokudo.identifier import BBIdentifier
from robokudo.pipeline import Pipeline
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_modification import (
    AddKinematicStructureEntityModification,
)

import robokudo.scripts.query_test_client


class QueryWorkerThread(threading.Thread):
    def __init__(self, node):
        threading.Thread.__init__(self)
        self.node = node
        self.result = None

    def run(self):
        self.result = query_simple_pipeline(self.node)


def query_simple_pipeline(node):
    cr_fr_config = CollectionReaderDescriptorFactory.create_descriptor(
        "file_reader",
        loop=True,
        target_dir=robokudo.utils.data_downloader.test_data_path() / Path("data"),
        kinect_height_fix_mode=True,
        color2depth_ratio=(0.5, 0.5),
    )

    # Restrict FOV of pointcloud to robustly get only one object
    pc_crop_config = PointcloudCropAnnotator.Descriptor()
    pc_crop_config.parameters.min_x = -0.3
    pc_crop_config.parameters.max_x = 0.3

    seq = Pipeline("TestPipeline")
    seq.add_children(
        [
            robokudo.idioms.pipeline_init(),
            QueryAnnotator(),
            CollectionReaderAnnotator(descriptor=cr_fr_config),
            ImagePreprocessorAnnotator("ImagePreprocessor"),
            PointcloudCropAnnotator(descriptor=pc_crop_config),
            PlaneAnnotator(),
            PointCloudClusterExtractor(),
            GenerateQueryResult(),
        ]
    )

    tree_result = robokudo.utils.tree_execution.run_tree_once(
        tree=seq, node=node, max_iterations=20, tick_rate=5
    )

    return seq, tree_result


@pytest.fixture
def tree_run(node):
    print("tree_run")
    return query_simple_pipeline(node)


class TestQueryInterface:
    def test_query_annotator_returns_running(self, tree_run):
        seq, tree_result = tree_run

        # We are not sending a query just yet. So the QueryAnnotator should block indefinitely with RUNNING
        assert tree_result is py_trees.common.Status.RUNNING

    def test_query(self, node):
        # seq, tree_result = query_simple_pipeline(node)
        multiprocessing.set_start_method("spawn", force=True)

        print("Starting Client Process")
        client_timeout_seconds = 20.0
        client_results = multiprocessing.Queue()
        client_readiness = multiprocessing.Event()
        client_process = multiprocessing.Process(
            target=robokudo.scripts.query_test_client.main,
            kwargs={
                "timeout_seconds": client_timeout_seconds,
                "result": client_results,
                "readiness": client_readiness,
            },
        )
        client_process.start()
        assert client_readiness.wait(timeout=client_timeout_seconds)

        print("Starting QueryWorker Thread")
        # Start the ActionServer/Query Interface PPT after its client is ready.
        worker_thread = QueryWorkerThread(node)
        worker_thread.start()

        worker_thread.join()
        print("Thread joined")
        seq, tree_result = worker_thread.result
        client_process.join()
        print("Process joined")

        # PPT with action server completed successfully
        assert tree_result is py_trees.common.Status.SUCCESS
        # The action client did not time out
        client_result = client_results.get()
        assert client_result["timed_out"] is False
        assert client_result["goal_status"] is GoalStatus.STATUS_SUCCEEDED
        assert len(client_result["goal_result"].res) == 1


# %% the belief state changes in the query result


class TestQueryResultCarriesBeliefStateChanges:
    def test_text_result_holds_the_objects_the_run_added(self):
        generate_query_result = GenerateQueryResult()
        pipeline = Pipeline("QueryResultTestPipeline")
        pipeline.add_child(generate_query_result)
        world = BeliefState.current().world
        perceived_body = Body(name=PrefixedName("perceived_object"))
        with world.modify_world():
            world.add_body(perceived_body)

        generate_query_result.update()

        answer = Blackboard().get(BBIdentifier.QUERY_ANSWER)
        update = from_json(
            json.loads(answer.text_result),
            **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
        )
        assert [
            modification.kinematic_structure_entity.id
            for modification in update.modification_block.modifications
            if isinstance(modification, AddKinematicStructureEntityModification)
        ] == [perceived_body.id]
