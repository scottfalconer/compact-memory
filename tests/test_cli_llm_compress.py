import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

# Adjust sys.path if compact_memory is not installed in a way that tests can find it
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from compact_memory.main import app # Main Typer application
from compact_memory.engines.base import BaseCompressionEngine, CompressedMemory
from compact_memory.llm_providers_abc import LLMProvider
from compact_memory.engines.registry import register_engine, unregister_engine, get_engine_metadata, _ENGINES
from compact_memory.config import LLM_MODELS_CONFIG_PATH as DEFAULT_LLM_MODELS_CONFIG_PATH

# --- DummyLLMEngine for Testing ---
class DummyLLMEngine(BaseCompressionEngine):
    id = "dummy_llm_engine"
    requires_llm = True  # Key attribute

    def __init__(self, config: Optional[dict] = None, llm_provider: Optional[LLMProvider] = None):
        super().__init__(config=config)
        self.llm_provider = llm_provider
        self.engine_config = config if config else {}
        if llm_provider is None:
            # This check might be too strict if an engine could operate without LLM but optionally use one.
            # For this dummy, we enforce it to test LLM provision.
            raise ValueError("DummyLLMEngine requires an LLMProvider for this test setup.")

        # Store received config for assertions in output
        self.received_gist_model = self.engine_config.get("gist_model_name")
        self.received_gist_length = self.engine_config.get("gist_length")
        self.received_qa_model = self.engine_config.get("qa_model_name") # Added for completeness

    def compress(self, text: str, budget: int, **kwargs) -> CompressedMemory:
        summary = f"LLM summary of '{text[:20]}' with provider {type(self.llm_provider).__name__}."
        if self.received_gist_model:
            summary += f" GistModel: {self.received_gist_model}."
        if self.received_gist_length:
            summary += f" GistLength: {self.received_gist_length}."
        if self.received_qa_model:
            summary += f" QAModel: {self.received_qa_model}."
        return CompressedMemory(text=summary, original_length=len(text), compressed_length=len(summary))

# --- Test Class ---
class TestCliLLMCompress(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.test_dir = Path("test_cli_temp_data")
        self.test_dir.mkdir(exist_ok=True)
        self.input_file = self.test_dir / "input.txt"
        with open(self.input_file, "w") as f:
            f.write("This is some sample text for compression.")

        # Register the dummy engine before each test
        # Store original state of dummy_llm_engine if it somehow existed
        self._original_dummy_engine_meta = _ENGINES.get(DummyLLMEngine.id)
        register_engine(DummyLLMEngine)

        # Handle llm_models_config.yaml
        # We'll create it in test_dir to avoid interfering with a real one
        self.temp_llm_config_path = self.test_dir / "llm_models_config.yaml"

        # Backup and restore default llm_models_config.yaml path for test isolation
        self.original_llm_models_config_path_in_module = None
        if 'compact_memory.config' in sys.modules:
            self.original_llm_models_config_path_in_module = getattr(sys.modules['compact_memory.config'], 'LLM_MODELS_CONFIG_PATH', None)

        # For tests that rely on a specific llm_models_config.yaml, we'll point the module-level variable to our test one.
        # And ensure Config instances created during the test use this path.
        # This is done more granularly in tests needing it via @mock.patch.

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

        # Unregister dummy engine and restore original state if any
        unregister_engine(DummyLLMEngine.id)
        if self._original_dummy_engine_meta:
             _ENGINES[DummyLLMEngine.id] = self._original_dummy_engine_meta

        if self.original_llm_models_config_path_in_module and 'compact_memory.config' in sys.modules:
            setattr(sys.modules['compact_memory.config'], 'LLM_MODELS_CONFIG_PATH', self.original_llm_models_config_path_in_module)


    def _create_temp_llm_config(self, content: str):
        # This helper creates the config in the designated test temporary directory.
        # Tests will then use mocking to make the Config class load from this path.
        with open(self.temp_llm_config_path, "w") as f:
            f.write(content)
        return self.temp_llm_config_path

    # --- Test Cases ---

    @mock.patch('compact_memory.main.app_config') # Mocks the app_config used in ctx.obj
    def test_compress_llm_engine_no_llm_config_error(self, mock_app_config_in_main):
        # Test that an error occurs if an LLM-dependent engine is used without LLM config
        # Ensure the mocked app_config returns None for default_model_id
        mock_app_config_in_main.get.side_effect = lambda key, default=None: None if key == "default_model_id" else default
        mock_app_config_in_main.get_all_llm_configs.return_value = {} # No named configs available

        result = self.runner.invoke(
            app,
            [
                "compress", str(self.input_file),
                "--engine", DummyLLMEngine.id,
                "--budget", "50"
            ]
        )
        self.assertNotEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn(f"Error: Engine '{DummyLLMEngine.id}' requires an LLM.", result.stdout)

    @mock.patch('compact_memory.main.app_config') # Target the config instance the CLI command will use
    def test_compress_with_llm_config_file(self, mock_app_config_in_main):
        temp_config_file_path = self._create_temp_llm_config("my-test-mock: {provider: mock, model_name: mock-model-from-file}")

        # Configure the mocked app_config to use this temporary file
        # This requires that the app_config instance correctly re-initializes or its methods are individually mocked.
        # The factory `create_llm_provider` will receive this app_config.
        mock_app_config_in_main.get_llm_config.side_effect = lambda name: {"provider": "mock", "model_name": "mock-model-from-file"} if name == "my-test-mock" else None
        mock_app_config_in_main.get_all_llm_configs.return_value = {"my-test-mock": {"provider": "mock"}}
        # Ensure default_model_id is not used
        mock_app_config_in_main.get.side_effect = lambda key, default=None: None if key == "default_model_id" else default

        result = self.runner.invoke(
            app,
            [
                "compress", str(self.input_file),
                "--engine", DummyLLMEngine.id,
                "--llm-config", "my-test-mock",
                "--budget", "50"
            ]
        )
        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("LLM summary", result.stdout)
        self.assertIn("MockLLMProvider", result.stdout)
        self.assertIn("GistModel: mock-model-from-file", result.stdout)


    @mock.patch('compact_memory.main.app_config')
    def test_compress_with_direct_llm_flags_mock(self, mock_app_config_in_main):
        # Ensure default_model_id is not used
        mock_app_config_in_main.get.side_effect = lambda key, default=None: None if key == "default_model_id" else default
        mock_app_config_in_main.get_all_llm_configs.return_value = {}
        result = self.runner.invoke(
            app,
            [
                "compress", str(self.input_file),
                "--engine", DummyLLMEngine.id,
                "--llm-provider-type", "mock",
                "--llm-model-name", "some-mock-model",
                "--budget", "50"
            ]
        )
        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("LLM summary", result.stdout)
        self.assertIn("MockLLMProvider", result.stdout)
        self.assertIn("GistModel: some-mock-model", result.stdout)


    @mock.patch('compact_memory.main.app_config')
    def test_compress_mutual_exclusion_llm_config_and_provider_type(self, mock_app_config_in_main):
        # No need to create temp llm config file if we are just testing CLI flag validation
        mock_app_config_in_main.get_llm_config.return_value = {"provider": "mock"} # Simulate config exists

        result = self.runner.invoke(
            app,
            [
                "compress", str(self.input_file),
                "--engine", DummyLLMEngine.id,
                "--llm-config", "my-test-mock", # This implies a config file entry
                "--llm-provider-type", "mock",   # This is the conflicting direct flag
                "--llm-model-name", "any-model",
                "--budget", "50"
            ]
        )
        self.assertNotEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("Error: --llm-config cannot be used with --llm-provider-type", result.stdout)

    def test_compress_missing_model_name_with_provider_type(self):
        # This test doesn't need app_config mocking as it's pure CLI validation
        result = self.runner.invoke(
            app,
            [
                "compress", str(self.input_file),
                "--engine", DummyLLMEngine.id,
                "--llm-provider-type", "mock", # Missing --llm-model-name
                "--budget", "50"
            ]
        )
        self.assertNotEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("Error: --llm-model-name must be provided if --llm-provider-type is specified", result.stdout)

    @mock.patch('compact_memory.main.app_config')
    def test_compress_readagent_flags_override(self, mock_app_config_in_main):
        mock_app_config_in_main.get.side_effect = lambda key, default=None: None if key == "default_model_id" else default
        mock_app_config_in_main.get_all_llm_configs.return_value = {}


        result = self.runner.invoke(
            app,
            [
                "compress", str(self.input_file),
                "--engine", DummyLLMEngine.id,
                "--llm-provider-type", "mock",
                "--llm-model-name", "default_mock_model",
                "--readagent-gist-model-name", "override_gist_model",
                "--readagent-gist-length", "123",
                "--readagent-qa-model-name", "override_qa_model",
                "--budget", "50"
            ]
        )
        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("LLM summary", result.stdout)
        self.assertIn("MockLLMProvider", result.stdout)
        self.assertIn("GistModel: override_gist_model", result.stdout)
        self.assertIn("GistLength: 123", result.stdout)
        self.assertIn("QAModel: override_qa_model", result.stdout)

    @mock.patch('compact_memory.main.app_config')
    def test_compress_readagent_flags_default_from_llm_model_name(self, mock_app_config_in_main):
        mock_app_config_in_main.get.side_effect = lambda key, default=None: None if key == "default_model_id" else default
        mock_app_config_in_main.get_all_llm_configs.return_value = {}

        result = self.runner.invoke(
            app,
            [
                "compress", str(self.input_file),
                "--engine", DummyLLMEngine.id,
                "--llm-provider-type", "mock",
                "--llm-model-name", "this_is_my_main_model",
                "--readagent-gist-length", "99",
                "--budget", "50"
            ]
        )
        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("LLM summary", result.stdout)
        self.assertIn("MockLLMProvider", result.stdout)
        self.assertIn("GistModel: this_is_my_main_model", result.stdout)
        self.assertIn("GistLength: 99", result.stdout)
        self.assertIn("QAModel: this_is_my_main_model", result.stdout)

    @mock.patch('compact_memory.main.app_config')
    def test_compress_fallback_to_default_model_id_as_provider_model(self, mock_app_config_in_main):
        # Simulate Config.get('default_model_id') returning "mock/my-default-mock-model"
        mock_app_config_in_main.get.side_effect = lambda key, default=None: "mock/my-default-mock-model" if key == "default_model_id" else default
        mock_app_config_in_main.get_llm_config.return_value = None # No named config for this one
        mock_app_config_in_main.get_all_llm_configs.return_value = {}


        result = self.runner.invoke(
            app,
            [
                "compress", str(self.input_file),
                "--engine", DummyLLMEngine.id,
                "--budget", "50"
            ]
        )
        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("No LLM specified directly; using default from config: mock/my-default-mock-model", result.stdout)
        self.assertIn("MockLLMProvider", result.stdout)
        self.assertIn("GistModel: my-default-mock-model", result.stdout) # Check model from provider/model string

    @mock.patch('compact_memory.main.app_config')
    def test_compress_fallback_to_default_model_id_as_named_config(self, mock_app_config_in_main):
        temp_config_file_path = self._create_temp_llm_config("default-llm-name: {provider: mock, model_name: 'mock-from-named-default'}")

        # Configure the mock app_config to use this temporary file's content when queried
        mock_app_config_in_main.get.side_effect = lambda key, default=None: "default-llm-name" if key == "default_model_id" else default

        def get_llm_config_side_effect(name):
            if name == "default-llm-name":
                return {"provider": "mock", "model_name": "mock-from-named-default"}
            return None
        mock_app_config_in_main.get_llm_config.side_effect = get_llm_config_side_effect
        mock_app_config_in_main.get_all_llm_configs.return_value = {"default-llm-name": {"provider": "mock", "model_name": "mock-from-named-default"}}

        # To make the actual Config load this temporary file for the factory, we use the env var.
        # The factory `create_llm_provider` receives `app_config` (which is `mock_app_config_in_main`).
        # The factory will use this `app_config`'s methods first.
        # If `actual_llm_config_name` is determined (from default_model_id), it calls `app_config.get_llm_config(actual_llm_config_name)`.
        # This is already mocked above. So, env var might not be strictly needed here IF the `app_config` passed to factory is the one we mock.
        # The CLI command gets `app_config = ctx.obj['config']`. `compact_memory.main.app_config` is the source of this.

        result = self.runner.invoke(
            app,
            [
                "compress", str(self.input_file),
                "--engine", DummyLLMEngine.id,
                "--budget", "50",
            ],
             # The env var is crucial if the Config object in the factory needs to load the file itself,
             # but since we mock get_llm_config on the instance passed to the factory, it might be okay.
             # However, the factory itself might instantiate a new Config() if app_config=None.
             # The CLI passes the app_config from context, so that path is not taken.
            env={"COMPACT_MEMORY_LLM_MODELS_CONFIG_PATH": str(temp_config_file_path)} # Still good for safety.
        )

        self.assertEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("No LLM specified directly; using default from config: default-llm-name", result.stdout)
        self.assertIn("MockLLMProvider", result.stdout)
        self.assertIn("GistModel: mock-from-named-default", result.stdout)

    @mock.patch('compact_memory.main.app_config')
    def test_compress_error_if_default_model_id_invalid_format(self, mock_app_config_in_main):
        mock_app_config_in_main.get.side_effect = lambda key, default=None: "invalid_default/format/extra_slash" if key == "default_model_id" else default
        mock_app_config_in_main.get_llm_config.return_value = None
        mock_app_config_in_main.get_all_llm_configs.return_value = {}

        result = self.runner.invoke(
            app,
            [
                "compress", str(self.input_file),
                "--engine", DummyLLMEngine.id,
                "--budget", "50"
            ]
        )
        self.assertNotEqual(result.exit_code, 0, msg=result.stdout)
        self.assertIn("Error: Default model ID 'invalid_default/format/extra_slash' is not a valid", result.stdout)

if __name__ == '__main__':
    unittest.main()
