"""A thin gateway over two LLM providers, TogetherAI and Anthropic.

Dataquest guided project, Build a Multi-Provider LLM Gateway.

Calling code talks only to `generate()`, which takes the same arguments and
returns the same shape whatever provider handles the request:

    {"provider": ..., "model": ..., "text": ..., "usage": {...}}

Two things here go past the brief, both for the same reason: a gateway's entire
value is that the same input means the same thing on both providers, and that
claim cannot be checked by looking at two live responses.

**System messages are joined, not overwritten.** TogetherAI takes system
messages inside the `messages` array, so every system message is sent.
Anthropic takes a single top-level `system` field, and the obvious loop keeps
only the last one. Give both providers the same two system messages and
TogetherAI follows both while Anthropic silently follows one. That is the
gateway failing at the one job it has. This joins them with a blank line so
both providers receive the same instruction set, and `system_divergence()`
demonstrates the difference against the naive approach.

**Rate limits are retried.** A 429 means the request was fine and merely early,
so it is retried with exponential backoff. Every other error is raised at once,
because retrying a bad key or a malformed payload never helps.

The API key is read from the environment and never logged. Error messages pass
through `_redact()` so a key echoed back by a provider does not end up in a
traceback or a CI log.
"""

import os
import time

import requests

from constants import (ANTHROPIC_KEY_ENV, ANTHROPIC_URL, ANTHROPIC_VERSION,
                       BACKOFF_FACTOR, DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT,
                       INITIAL_BACKOFF, MAX_RETRIES, SUPPORTED_PROVIDERS,
                       TOGETHER_KEY_ENV, TOGETHER_URL)


class LLMGatewayError(Exception):
    """Anything that goes wrong inside the gateway.

    A named exception gives callers one thing to catch and makes it obvious the
    failure came from this module rather than from somewhere inside requests.
    """


def _redact(text, *secrets):
    """Remove any API key from a string before it reaches a log or traceback.

    Providers sometimes echo part of a request back in an error body, and error
    strings end up in CI output where they are durable and public.
    """
    cleaned = str(text)
    for secret in secrets:
        if secret and len(secret) > 6:
            cleaned = cleaned.replace(secret, "[redacted]")
    return cleaned


def _require_key(variable):
    key = os.getenv(variable)
    if not key:
        raise LLMGatewayError(
            "%s is not set. Export it before calling the gateway." % variable)
    return key


def _post(url, headers, payload, timeout, secret, session=None):
    """POST with backoff on 429, and nothing else retried.

    Returns the parsed JSON body. Raises LLMGatewayError on any failure, with
    the key scrubbed out of the message.
    """
    http = session or requests
    backoff = INITIAL_BACKOFF
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = http.post(url, headers=headers, json=payload,
                                 timeout=timeout)
        except requests.exceptions.RequestException as problem:
            last_error = LLMGatewayError(
                "Request to %s failed: %s" % (url, _redact(problem, secret)))
            time.sleep(backoff)
            backoff *= BACKOFF_FACTOR
            continue

        if response.status_code == 429:
            last_error = LLMGatewayError(
                "Rate limited by %s after %d attempts." % (url, MAX_RETRIES))
            wait = response.headers.get("Retry-After")
            try:
                wait = float(wait) if wait is not None else backoff
            except (TypeError, ValueError):
                wait = backoff
            time.sleep(wait)
            backoff *= BACKOFF_FACTOR
            continue

        if response.status_code != 200:
            raise LLMGatewayError(
                "%s returned HTTP %d: %s"
                % (url, response.status_code,
                   _redact(response.text[:300], secret)))

        try:
            return response.json()
        except ValueError:
            raise LLMGatewayError(
                "%s returned a non-JSON body: %s"
                % (url, _redact(response.text[:200], secret)))

    raise last_error or LLMGatewayError("Request to %s failed." % url)


def split_system_messages(messages):
    """Separate system messages from the conversation.

    Returns (system_text, other_messages). Every system message is kept and
    joined with a blank line, rather than the last one winning. See the module
    docstring for why that matters.
    """
    system_parts = []
    others = []
    for message in messages or []:
        if message.get("role") == "system":
            content = message.get("content")
            if content:
                system_parts.append(content)
        else:
            others.append(message)
    return "\n\n".join(system_parts), others


def system_divergence(messages):
    """What the naive loop would send to Anthropic versus what this sends.

    Returns (naive, joined). When they differ, the naive version is dropping
    instructions that TogetherAI would have received, which is the gateway
    quietly meaning two different things on two providers.
    """
    naive = None
    for message in messages or []:
        if message.get("role") == "system":
            naive = message.get("content")
    joined, _ = split_system_messages(messages)
    return naive, joined


def _call_together(model, messages, max_tokens, timeout=DEFAULT_TIMEOUT,
                   session=None):
    """Call TogetherAI's chat completions endpoint and return the text."""
    key = _require_key(TOGETHER_KEY_ENV)
    headers = {
        "Authorization": "Bearer %s" % key,
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    data = _post(TOGETHER_URL, headers, payload, timeout, key, session)

    # .get() throughout: a provider response is not guaranteed to match its
    # own documentation, and a KeyError here tells the caller nothing useful.
    choices = data.get("choices", [])
    if not choices:
        raise LLMGatewayError(
            "TogetherAI returned no choices. Keys present: %s"
            % sorted(data.keys()))
    text = choices[0].get("message", {}).get("content")
    if text is None:
        raise LLMGatewayError(
            "TogetherAI returned a choice with no message content.")
    return text, _together_usage(data)


def _call_anthropic(model, messages, max_tokens, timeout=DEFAULT_TIMEOUT,
                    session=None):
    """Call Anthropic's messages endpoint and return the text."""
    key = _require_key(ANTHROPIC_KEY_ENV)
    headers = {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }
    system_text, conversation = split_system_messages(messages)
    payload = {
        "model": model,
        "messages": conversation,
        "max_tokens": max_tokens,
    }
    if system_text:
        payload["system"] = system_text

    data = _post(ANTHROPIC_URL, headers, payload, timeout, key, session)

    blocks = data.get("content", [])
    # The first block is not guaranteed to be text: Anthropic can return
    # tool_use and other block types, so take the first text block rather than
    # assuming position zero.
    text_block = next(
        (block for block in blocks if block.get("type") == "text"), None)
    if text_block is None:
        kinds = [block.get("type") for block in blocks]
        raise LLMGatewayError(
            "Anthropic returned no text block. Block types: %s" % kinds)
    text = text_block.get("text")
    if text is None:
        raise LLMGatewayError("Anthropic text block has no text field.")
    return text, _anthropic_usage(data)


def _together_usage(data):
    """Normalize TogetherAI's token counts onto one shape."""
    usage = data.get("usage") or {}
    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
    }


def _anthropic_usage(data):
    """Normalize Anthropic's token counts onto the same shape.

    The two providers name these differently, so a caller that wanted to log
    or bill on token counts would otherwise still need provider-specific code,
    which is the thing the gateway exists to remove.
    """
    usage = data.get("usage") or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


def generate(provider, model, messages, max_tokens=DEFAULT_MAX_TOKENS,
             timeout=DEFAULT_TIMEOUT, session=None):
    """Send a conversation to a provider and get a normalized result back.

    Returns {"provider", "model", "text", "usage"}. Callers never see a
    provider-shaped response.
    """
    if not messages:
        raise LLMGatewayError("messages must not be empty.")
    if not isinstance(messages, (list, tuple)):
        raise LLMGatewayError("messages must be a list of role/content dicts.")
    for message in messages:
        if not isinstance(message, dict) or "role" not in message:
            raise LLMGatewayError(
                "Each message needs a 'role'; got %r" % (message,))

    key = str(provider).strip().lower() if provider else ""
    if key == "together":
        text, usage = _call_together(model, messages, max_tokens, timeout,
                                     session)
    elif key == "anthropic":
        text, usage = _call_anthropic(model, messages, max_tokens, timeout,
                                      session)
    else:
        raise LLMGatewayError(
            "Unsupported provider %r. Supported: %s"
            % (provider, ", ".join(SUPPORTED_PROVIDERS)))

    return {
        "provider": key,
        "model": model,
        "text": text,
        "usage": usage,
    }
