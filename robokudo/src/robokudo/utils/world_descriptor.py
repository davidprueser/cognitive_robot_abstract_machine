"""
World descriptor utilities for Robokudo.
"""

from __future__ import annotations

from typing_extensions import TYPE_CHECKING

import robokudo.utils.module_loader as module_loader
from robokudo.exceptions import WorldDescriptorLoadError

if TYPE_CHECKING:
    from robokudo.annotators.world_descriptor_bootstrap import (
        WorldDescriptorBootstrapAnnotator,
    )
    from robokudo.world_descriptor import BaseWorldDescriptor


def load_world_descriptor(
    annotator: WorldDescriptorBootstrapAnnotator,
) -> BaseWorldDescriptor:
    """
    Load the world descriptor the annotator is configured with.
    """
    ros_package = annotator.world_descriptor_ros_package
    module_name = annotator.world_descriptor_name
    loader = module_loader.ModuleLoader()
    try:
        return loader.load_world_descriptor(ros_package, module_name)
    except Exception as error:
        raise WorldDescriptorLoadError(
            ros_package=ros_package,
            module_name=module_name,
        ) from error
