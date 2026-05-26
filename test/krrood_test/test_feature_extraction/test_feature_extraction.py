import os

import numpy as np
import plotly.graph_objects
from sqlalchemy.orm import Session

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import (
    underspecified,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from krrood.parametrization.feature_extractor import FeatureExtractor
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.relational.learn_rspn import (
    learn_probabilistic_circuit,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from semantic_digital_twin_test.test_adapters.test_environment_generation import (
    query_for_shelves,
)
from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import (
    NestedAction,
    KRROODPose,
    KRROODPosition,
    KRROODOrientation,
    SceneRoom,
    SceneObject,
    SceneObjectType,
)
from ..dataset.ormatic_interface import Base
from ..dataset.semantic_world_like_classes import Body


def test_features_extraction():
    action = underspecified(NestedAction)(
        pose=underspecified(KRROODPose)(
            position=underspecified(KRROODPosition)(x=2.0, y=..., z=...),
            orientation=underspecified(KRROODOrientation)(x=..., y=..., z=..., w=...),
        ),
        obj=Body(name="body"),
    )

    parameters = UnderspecifiedParameters(action)
    fully_factorized_circuit = fully_factorized(parameters.variables.values())
    assert len(parameters.truncation_assignments_from_krrood_variables) == 0

    probabilistic_registry = DictRegistry({NestedAction: fully_factorized_circuit})

    np.random.seed(69)
    backend = ProbabilisticBackend(probabilistic_registry, number_of_samples=50)
    samples = list(backend.evaluate(action))

    assert all(
        [sample.pose.position.x == samples[0].pose.position.x for sample in samples]
    )
    samples_to_daos = [to_dao(sample) for sample in samples]

    feature_extractor = FeatureExtractor.from_instances(samples_to_daos)
    dataframe = feature_extractor.create_dataframe(samples_to_daos)

    assert [
        dataframe[column].dtype in (np.float64, np.int64)
        for column in dataframe.columns
    ]
    assert dataframe.shape == (len(samples_to_daos), len(feature_extractor.features))


def test_feature_extraction_with_aggregations():

    objects = [
        SceneObject(type=SceneObjectType.TABLE),
        SceneObject(type=SceneObjectType.CHAIR),
        SceneObject(type=SceneObjectType.CHAIR),
        SceneObject(type=SceneObjectType.CHAIR),
    ]
    chair_objects = [obj for obj in objects if obj.type == SceneObjectType.CHAIR]
    room = SceneRoom(
        position=KRROODPosition(x=2.0, y=1.0, z=0.0),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=objects[:3],
    )
    room2 = SceneRoom(
        position=KRROODPosition(x=4.0, y=3.0, z=0.0),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=objects,
    )
    room_dao = to_dao(room)
    room2_dao = to_dao(room2)
    feature_extractor = FeatureExtractor.from_instances([room_dao])

    # print(feature_extractor.features)
    # assert (
    #     len(feature_extractor.features) == 9
    # )  # position: 3, orientation: 4, objects: 1 (count for chair, count for table)
    # assert len(chair_objects) in feature_extractor.apply_mapping(room_dao)

    mapping = feature_extractor.apply_mapping(room_dao)
    rpc = RelationalProbabilisticCircuit(SceneRoom)
    rpc.fit([room_dao, room2_dao], feature_extractor)

    room_query = underspecified(SceneRoom)(
        position=underspecified(KRROODPosition)(x=..., y=..., z=...),
        orientation=underspecified(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[underspecified(SceneObject)(type=...) for _ in range(4)],
    )
    room_query.resolve()
    model = rpc.ground(room_query)
    print(model)
