import logging
from pathlib import Path

import experiments
import semantic_digital_twin.orm.ormatic_interface
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.utils import classes_of_module
import experiments.scene_generation_experiments.data_preprocessing

ignored_classes = set(classes_of_module(experiments.scene_generation_experiments.data_preprocessing))

# Create an ORMatic object with the classes to be mapped
ormatic = ORMatic.from_package(
    [experiments], [semantic_digital_twin.orm.ormatic_interface], ignored_classes, type_mappings={}
)
logging.getLogger("krrood").setLevel(logging.DEBUG)

# Generate the ORM classes
ormatic.make_all_tables()

ormatic_interface_path = (
    Path(__file__).parent.parent
    / "src"
    / "experiments"
    / "orm"
    / "ormatic_interface.py"
)
with open(ormatic_interface_path, "w") as f:
    ormatic.to_sqlalchemy_file(f)
