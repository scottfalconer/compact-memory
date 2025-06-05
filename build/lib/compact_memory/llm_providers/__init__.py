"""LLM provider implementations.

The optional providers are imported lazily so missing heavy dependencies do not
break basic functionality or tests that don't require them.
"""

from .openai_provider import OpenAIProvider

__all__ = ["OpenAIProvider"]

try:  # Gemini provider requires ``google-generativeai``
    from .gemini_provider import GeminiProvider
except Exception:  # pragma: no cover - optional dependency may be missing
    GeminiProvider = None  # type: ignore
else:  # pragma: no cover - imported when dependency is available
    __all__.append("GeminiProvider")

try:  # Local provider requires transformers
    from .local_provider import LocalTransformersProvider
except Exception:  # pragma: no cover - optional dependency may be missing
    LocalTransformersProvider = None  # type: ignore
else:  # pragma: no cover - imported when dependency is available
    __all__.append("LocalTransformersProvider")
