import os
import unittest
from unittest import mock
import sys

# Ensure compact_memory is discoverable if tests are run directly
# This might not be needed if using pytest and proper project structure/installation
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from compact_memory.llm_providers.factory import create_llm_provider
from compact_memory.llm_providers_abc import LLMProvider
from compact_memory.llm_providers import OpenAIProvider, MockLLMProvider
from compact_memory.config import Config

# Attempt to import optional providers for more complete testing if available
try:
    from compact_memory.llm_providers import LocalTransformersProvider
except ImportError:
    LocalTransformersProvider = None

try:
    from compact_memory.llm_providers import GeminiProvider
except ImportError:
    GeminiProvider = None


class TestLLMProviderFactory(unittest.TestCase):

    def setUp(self):
        # Store and clear relevant environment variables before each test
        self.original_openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.original_google_api_key = os.environ.get("GOOGLE_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

    def tearDown(self):
        # Restore original environment variables
        if self.original_openai_api_key is not None:
            os.environ["OPENAI_API_KEY"] = self.original_openai_api_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        if self.original_google_api_key is not None:
            os.environ["GOOGLE_API_KEY"] = self.original_google_api_key
        elif "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

    @mock.patch.object(Config, "get_llm_config")
    @mock.patch.object(Config, "get_all_llm_configs")
    def test_create_with_named_config_mock(self, mock_get_all_configs, mock_get_config):
        mock_get_config.return_value = {"provider": "mock"}
        provider = create_llm_provider(config_name="my-mock-config", app_config=Config())
        self.assertIsInstance(provider, MockLLMProvider)
        mock_get_config.assert_called_with("my-mock-config")

    @mock.patch.object(Config, "get_llm_config")
    @mock.patch.object(Config, "get_all_llm_configs")
    def test_create_with_named_config_openai(self, mock_get_all_configs, mock_get_config):
        mock_get_config.return_value = {"provider": "openai", "api_key": "fake_key_from_config"}
        provider = create_llm_provider(config_name="my-openai-config", app_config=Config())
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), "fake_key_from_config")

    @mock.patch.object(Config, "get_llm_config")
    @mock.patch.object(Config, "get_all_llm_configs")
    def test_create_with_named_config_not_found(self, mock_get_all_configs, mock_get_config):
        mock_get_config.return_value = None
        mock_get_all_configs.return_value = {"existing-config": {}}
        with self.assertRaisesRegex(ValueError, "LLM configuration 'nonexistent-config' not found"):
            create_llm_provider(config_name="nonexistent-config", app_config=Config())

    def test_create_with_direct_provider_mock(self):
        provider = create_llm_provider(provider_type="mock")
        self.assertIsInstance(provider, MockLLMProvider)

    def test_create_with_direct_provider_openai(self):
        provider = create_llm_provider(provider_type="openai", api_key="direct_test_key")
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), "direct_test_key")

    def test_create_with_unknown_provider_type(self):
        with self.assertRaisesRegex(ValueError, "Unsupported LLM provider type: 'unknown_provider'"):
            create_llm_provider(provider_type="unknown_provider")

    def test_api_key_handling_openai_from_config_overrides_direct_if_config_name_passed(self):
        # If config_name is passed, its api_key should be used, even if api_key param is also passed.
        # The factory logic prioritizes values from the named config.
        with mock.patch.object(Config, 'get_llm_config', return_value={'provider': 'openai', 'api_key': 'config_key'}):
            provider = create_llm_provider(config_name="some-config", api_key="direct_key_ignored", app_config=Config())
            self.assertIsInstance(provider, OpenAIProvider)
            self.assertEqual(os.environ.get("OPENAI_API_KEY"), "config_key")

    def test_api_key_handling_openai_direct_used_if_no_config_name(self):
        provider = create_llm_provider(provider_type="openai", api_key="direct_key_used")
        self.assertIsInstance(provider, OpenAIProvider)
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), "direct_key_used")


    def test_value_error_if_no_provider_type_determinable(self):
        # No config_name, no provider_type
        with self.assertRaisesRegex(ValueError, "LLM provider type must be specified"):
            create_llm_provider(app_config=Config()) # Pass empty config

    @unittest.skipIf(LocalTransformersProvider is None, "LocalTransformersProvider not available (transformers/torch probably missing)")
    def test_create_local_provider_when_available(self):
        provider = create_llm_provider(provider_type="local")
        self.assertIsInstance(provider, LocalTransformersProvider)

    @mock.patch('compact_memory.llm_providers.factory.LocalTransformersProvider', None)
    def test_create_local_provider_import_error_if_set_to_none(self):
        # This simulates LocalTransformersProvider being None in factory's scope due to initial import error in __init__.py
        with self.assertRaisesRegex(ImportError, "Local LLM provider \(LocalTransformersProvider\) could not be imported.*LocalTransformersProvider is None"):
            create_llm_provider(provider_type="local")

    @mock.patch.dict(sys.modules, {'compact_memory.llm_providers.local_provider': None})
    def test_create_local_provider_import_error_on_module_level(self):
        # This simulates the local_provider module itself not being importable
         with self.assertRaisesRegex(ImportError, "Local LLM provider \(LocalTransformersProvider\) could not be imported."):
            create_llm_provider(provider_type="local")


    @unittest.skipIf(GeminiProvider is None, "GeminiProvider not available (google-generativeai probably missing)")
    def test_create_gemini_provider_when_available(self):
        provider = create_llm_provider(provider_type="gemini", api_key="gemini_test_key")
        self.assertIsInstance(provider, GeminiProvider)
        self.assertEqual(os.environ.get("GOOGLE_API_KEY"), "gemini_test_key")

    @mock.patch('compact_memory.llm_providers.factory.GeminiProvider', None)
    def test_create_gemini_provider_import_error_if_set_to_none(self):
        with self.assertRaisesRegex(ImportError, "Gemini LLM provider \(GeminiProvider\) could not be imported.*GeminiProvider is None"):
            create_llm_provider(provider_type="gemini")

    @mock.patch.dict(sys.modules, {'compact_memory.llm_providers.gemini_provider': None})
    def test_create_gemini_provider_import_error_on_module_level(self):
         with self.assertRaisesRegex(ImportError, "Gemini LLM provider \(GeminiProvider\) could not be imported."):
            create_llm_provider(provider_type="gemini")

if __name__ == '__main__':
    unittest.main()
