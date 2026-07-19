class LLMError(Exception):
    """Base class for all LLM provider errors."""


class TransientError(LLMError):
    """Worth retrying the same provider (timeout, 5xx, network blip)."""


class RateLimitError(TransientError):
    """Provider is rate-limiting us; retry (with backoff) or move to the next provider."""


class FatalError(LLMError):
    """Retrying the same provider won't help (bad api key, invalid request, ...)."""
