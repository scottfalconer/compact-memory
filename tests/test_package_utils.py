from pathlib import Path

from compact_memory.package_utils import ( # Assuming this path and functions will be updated if they are strategy-specific
    load_engine_class_from_module, # Updated function name
    validate_package_dir, # This function might need internal updates for engine packages
    check_requirements_installed,
)
from CompressionEngine.core.engines_abc import ( # Updated path
    CompressionEngine, # Updated class name
    CompressedMemory,
    CompressionTrace,
)


def _write_engine_module(path: Path) -> None: # Updated function name
    path.write_text(
        """
from CompressionEngine.core.engines_abc import ( # Updated path
    CompressionEngine, # Updated class name
    CompressedMemory,
    CompressionTrace,
)

class DummyEngine(CompressionEngine): # Updated class name
    id = 'dummy_module_engine' # Updated id

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        return CompressedMemory(text='ok'), CompressionTrace(
            engine_name=self.id, # Updated parameter name
            engine_params={}, # Updated parameter name
            input_summary={},
            steps=[],
        )
"""
    )


def test_load_engine_class_from_module(tmp_path: Path) -> None: # Updated function name
    module_file = tmp_path / "engine.py" # Updated filename
    _write_engine_module(module_file) # Updated function call
    cls = load_engine_class_from_module(str(module_file), "DummyEngine") # Updated function call and class name
    assert issubclass(cls, CompressionEngine) # Updated class name
    inst = cls()
    mem, trace = inst.compress("text", 10)
    assert isinstance(mem, CompressedMemory)
    assert trace.engine_name == "dummy_module_engine" # Updated parameter name and value


def test_validate_package_dir(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    module_file = pkg / "engine.py" # Updated filename
    _write_engine_module(module_file) # Updated function call
    manifest = pkg / "engine_package.yaml" # Updated filename
    manifest.write_text(
        """
package_format_version: '1.0'
engine_id: dummy_module_engine # Updated field name and value
engine_class_name: DummyEngine # Updated field name and value
engine_module: engine # Updated field name and value
display_name: Dummy Engine Package # Updated value
version: '0.1'
authors: ['Tester']
description: Test engine package # Updated value
"""
    )
    errors, warnings = validate_package_dir(pkg) # validate_package_dir might need internal changes for engine packages
    assert errors == []
    assert "requirements.txt not found" in warnings


def test_check_requirements_installed(tmp_path: Path) -> None:
    req = tmp_path / "requirements.txt"
    req.write_text("nonexistent_pkg_xyz>=1.0")
    missing = check_requirements_installed(req)
    assert missing == ["nonexistent_pkg_xyz"]
