"""Configuration for the LLM gateway.

Everything the gateway needs to know about the outside world lives here: where
to send requests, which API version to target, and the defaults. Keeping it in
one file means an endpoint change is a one line edit rather than a hunt through
the logic.

Values come from each provider's API documentation.
"""

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_MAX_TOKENS = 300
DEFAULT_TIMEOUT = 30

# Retry settings for rate limiting. A 429 is the one status worth retrying:
# it means the request was well formed and simply arrived too soon.
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0
BACKOFF_FACTOR = 2.0

SUPPORTED_PROVIDERS = ("together", "anthropic")

# Environment variable names, kept here so nothing else has to spell them out.
TOGETHER_KEY_ENV = "TOGETHER_API_KEY"
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
