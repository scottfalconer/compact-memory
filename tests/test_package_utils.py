from pathlib import Path

from compact_memory.package_utils import (
    load_strategy_class_from_module,
    validate_package_dir,
    check_requirements_installed,
)
from compact_memory.engines import (
    BaseCompressionEngine as CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
)


def _write_strategy_module(path: Path) -> None:
    path.write_text(
        """
from compact_memory.engines import (
    BaseCompressionEngine as CompressionStrategy,
    CompressedMemory,
    CompressionTrace,
)

class DummyStrategy(CompressionStrategy):
    id = 'dummy_module'

    def compress(self, text_or_chunks, llm_token_budget, **kwargs):
        return CompressedMemory(text='ok'), CompressionTrace(
            strategy_name=self.id,
            strategy_params={},
            input_summary={},
            steps=[],
        )
"""
    )


def test_load_strategy_class_from_module(tmp_path: Path) -> None:
    module_file = tmp_path / "strategy.py"
    _write_strategy_module(module_file)
    cls = load_strategy_class_from_module(str(module_file), "DummyStrategy")
    assert issubclass(cls, CompressionStrategy)
    inst = cls()
    mem, trace = inst.compress("text", 10)
    assert isinstance(mem, CompressedMemory)
    assert trace.strategy_name == "dummy_module"


def test_validate_package_dir(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    module_file = pkg / "strategy.py"
    _write_strategy_module(module_file)
    manifest = pkg / "strategy_package.yaml"
    manifest.write_text(
        """
package_format_version: '1.0'
strategy_id: dummy_module
strategy_class_name: DummyStrategy
strategy_module: strategy
display_name: Dummy Package
version: '0.1'
authors: ['Tester']
description: Test package
"""
    )
    errors, warnings = validate_package_dir(pkg)
    assert errors == []
    assert "requirements.txt not found" in warnings


def test_check_requirements_installed(tmp_path: Path) -> None:
    req = tmp_path / "requirements.txt"
    req.write_text("nonexistent_pkg_xyz>=1.0")
    missing = check_requirements_installed(req)
    assert missing == ["nonexistent_pkg_xyz"]
