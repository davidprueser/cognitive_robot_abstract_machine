from __future__ import annotations
import pytest
import rclpy

from robokudo.utils.module_loader import ModuleLoader


class TestUtilsModuleLoader(object):
    @pytest.fixture(autouse=True)
    def module_loader(self) -> ModuleLoader:
        return ModuleLoader()

    invalid_modules = [
        ("robokudo", "non_existent_module.py"),
        ("non_existent_package", "non_existent_module.py"),
    ]

    def test_load_analysis_engine(self, module_loader: ModuleLoader) -> None:
        assert module_loader.load_analysis_engine("robokudo", "demo.py")

    @pytest.mark.parametrize(["package_name", "module_name"], invalid_modules)
    def test_load_analysis_engine_invalid_input(
        self, module_loader: ModuleLoader, package_name: str, module_name: str
    ) -> None:
        assert pytest.raises(
            ImportError, module_loader.load_analysis_engine, package_name, module_name
        )

    def test_load_annotator(self, module_loader: ModuleLoader) -> None:
        assert module_loader.load_annotator("robokudo", "query.py")

    @pytest.mark.parametrize(["package_name", "module_name"], invalid_modules)
    def test_load_annotator_invalid_input(
        self, module_loader: ModuleLoader, package_name: str, module_name: str
    ) -> None:
        assert pytest.raises(
            ImportError, module_loader.load_annotator, package_name, module_name
        )

    def test_load_camera_config(self, module_loader: ModuleLoader) -> None:
        assert module_loader.load_camera_config("robokudo", "config_tiago.py")

    @pytest.mark.parametrize(["package_name", "module_name"], invalid_modules)
    def test_load_camera_config_invalid_input(
        self, module_loader: ModuleLoader, package_name: str, module_name: str
    ) -> None:
        assert pytest.raises(
            ImportError, module_loader.load_camera_config, package_name, module_name
        )

    def test_load_io(self, module_loader: ModuleLoader) -> None:
        assert module_loader.load_io("robokudo", "camera_interface.py")

    @pytest.mark.parametrize(["package_name", "module_name"], invalid_modules)
    def test_load_io_invalid_input(
        self, module_loader: ModuleLoader, package_name: str, module_name: str
    ) -> None:
        assert pytest.raises(
            ImportError, module_loader.load_io, package_name, module_name
        )

    def test_load_world_descriptor(self, module_loader: ModuleLoader) -> None:
        assert module_loader.load_world_descriptor("robokudo", "world_iai_kitchen20.py")

    @pytest.mark.parametrize(["package_name", "module_name"], invalid_modules)
    def test_load_world_descriptor_invalid_input(
        self, module_loader: ModuleLoader, package_name: str, module_name: str
    ) -> None:
        assert pytest.raises(
            ImportError, module_loader.load_world_descriptor, package_name, module_name
        )

    def test_load_tree_components(self, module_loader: ModuleLoader) -> None:
        assert module_loader.load_tree_components(
            "robokudo", "query_based_task_scheduler.py"
        )

    @pytest.mark.parametrize(["package_name", "module_name"], invalid_modules)
    def test_load_tree_components_invalid_input(
        self, module_loader: ModuleLoader, package_name: str, module_name: str
    ) -> None:
        assert pytest.raises(
            ImportError, module_loader.load_tree_components, package_name, module_name
        )

    def test_load_types(self, module_loader: ModuleLoader) -> None:
        assert module_loader.load_types("robokudo", "annotation.py")

    @pytest.mark.parametrize(["package_name", "module_name"], invalid_modules)
    def test_load_types_invalid_input(
        self, module_loader: ModuleLoader, package_name: str, module_name: str
    ) -> None:
        assert pytest.raises(
            ImportError, module_loader.load_types, package_name, module_name
        )

    def test_load_utils(self, module_loader: ModuleLoader) -> None:
        assert module_loader.load_utils("robokudo", "module_loader.py")

    @pytest.mark.parametrize(["package_name", "module_name"], invalid_modules)
    def test_load_utils_invalid_input(
        self, module_loader: ModuleLoader, package_name: str, module_name: str
    ) -> None:
        assert pytest.raises(
            ImportError, module_loader.load_utils, package_name, module_name
        )
