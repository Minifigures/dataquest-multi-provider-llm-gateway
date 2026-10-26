"""Tests for the multi-provider LLM gateway.

Run with: python3 -m unittest -v test_llm_gateway

No test makes a network call or needs an API key. The provider contracts are
pinned as fixtures taken from the response shapes in each provider's
documentation, which is the only way to check that two different APIs are
actually being normalized onto one shape. The lesson's own suggested test
("call both providers and see if text comes back") cannot catch a normalization
bug, because both providers returning some text looks identical to both
providers returning the right text.

time.sleep is stubbed so the backoff paths run instantly.
"""

import os
import unittest

import requests

import constants
import llm_gateway as gateway
from llm_gateway import LLMGatewayError

# Response shapes as each provider documents them.
TOGETHER_RESPONSE = {
    "id": "abc",
    "choices": [
        {"message": {"role": "assistant",
                     "content": "An API is a set of rules."}}
    ],
    "usage": {"prompt_tokens": 18, "completion_tokens": 9, "total_tokens": 27},
}

ANTHROPIC_RESPONSE = {
    "id": "msg_1",
    "type": "message",
    "content": [{"type": "text", "text": "An API is a set of rules."}],
    "usage": {"input_tokens": 18, "output_tokens": 9},
}

# Anthropic can lead with a non-text block, which the brief flags explicitly.
ANTHROPIC_TOOL_FIRST = {
    "id": "msg_2",
    "content": [
        {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}},
        {"type": "text", "text": "The answer is 42."},
    ],
    "usage": {"input_tokens": 5, "output_tokens": 3},
}

MESSAGES = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Explain what an API is in one sentence."},
]


class FakeResponse:
    def __init__(self, status_code, payload=None, text=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else str(payload)
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Returns queued responses in order and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": dict(headers or {}),
                           "payload": json, "timeout": timeout})
        if not self.responses:
            raise AssertionError("FakeSession ran out of responses")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class GatewayTestCase(unittest.TestCase):
    """Sets both keys and removes the real sleep so backoff runs instantly."""

    def setUp(self):
        self._saved_env = {}
        for name in (constants.TOGETHER_KEY_ENV, constants.ANTHROPIC_KEY_ENV):
            self._saved_env[name] = os.environ.get(name)
            os.environ[name] = "test-key-0123456789"
        self._saved_sleep = gateway.time.sleep
        self.slept = []
        gateway.time.sleep = lambda seconds: self.slept.append(seconds)

    def tearDown(self):
        gateway.time.sleep = self._saved_sleep
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class TestSystemMessages(unittest.TestCase):
    """The behaviour the gateway exists to get right."""

    def test_single_system_message_is_lifted_out(self):
        system, rest = gateway.split_system_messages(MESSAGES)
        self.assertEqual(system, "You are concise.")
        self.assertEqual(rest, [MESSAGES[1]])

    def test_every_system_message_is_kept(self):
        messages = [
            {"role": "system", "content": "You are concise."},
            {"role": "system", "content": "Answer in British English."},
            {"role": "user", "content": "Explain an API."},
        ]
        system, rest = gateway.split_system_messages(messages)
        self.assertIn("You are concise.", system)
        self.assertIn("Answer in British English.", system)
        self.assertEqual(len(rest), 1)

    def test_the_naive_loop_drops_an_instruction(self):
        """The bug this design avoids, stated as a failing comparison.

        TogetherAI gets both system messages because they stay in the array.
        The obvious Anthropic loop keeps only the last. Same call, two
        different behaviours, which is the gateway failing at its one job.
        """
        messages = [
            {"role": "system", "content": "You are concise."},
            {"role": "system", "content": "Answer in British English."},
            {"role": "user", "content": "Explain an API."},
        ]
        naive, joined = gateway.system_divergence(messages)
        self.assertEqual(naive, "Answer in British English.")
        self.assertNotIn("You are concise.", naive)
        self.assertIn("You are concise.", joined)

    def test_no_divergence_with_one_system_message(self):
        naive, joined = gateway.system_divergence(MESSAGES)
        self.assertEqual(naive, joined)

    def test_no_system_message_means_no_system_field(self):
        messages = [{"role": "user", "content": "Hi"}]
        system, rest = gateway.split_system_messages(messages)
        self.assertEqual(system, "")
        self.assertEqual(rest, messages)

    def test_empty_system_content_is_ignored(self):
        messages = [{"role": "system", "content": ""},
                    {"role": "user", "content": "Hi"}]
        system, _ = gateway.split_system_messages(messages)
        self.assertEqual(system, "")


class TestTogether(GatewayTestCase):

    def test_happy_path_returns_normalized_shape(self):
        session = FakeSession([FakeResponse(200, TOGETHER_RESPONSE)])
        result = gateway.generate("together", "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
                                  MESSAGES, session=session)
        self.assertEqual(result["provider"], "together")
        self.assertEqual(result["text"], "An API is a set of rules.")
        self.assertEqual(sorted(result), ["model", "provider", "text", "usage"])

    def test_request_is_addressed_and_authenticated_correctly(self):
        session = FakeSession([FakeResponse(200, TOGETHER_RESPONSE)])
        gateway.generate("together", "m", MESSAGES, session=session)
        call = session.calls[0]
        self.assertEqual(call["url"], constants.TOGETHER_URL)
        self.assertTrue(call["headers"]["Authorization"].startswith("Bearer "))
        self.assertNotIn("x-api-key", call["headers"])

    def test_system_message_stays_in_the_array(self):
        session = FakeSession([FakeResponse(200, TOGETHER_RESPONSE)])
        gateway.generate("together", "m", MESSAGES, session=session)
        payload = session.calls[0]["payload"]
        self.assertNotIn("system", payload)
        roles = [m["role"] for m in payload["messages"]]
        self.assertIn("system", roles)

    def test_max_tokens_defaults_and_overrides(self):
        session = FakeSession([FakeResponse(200, TOGETHER_RESPONSE),
                               FakeResponse(200, TOGETHER_RESPONSE)])
        gateway.generate("together", "m", MESSAGES, session=session)
        self.assertEqual(session.calls[0]["payload"]["max_tokens"],
                         constants.DEFAULT_MAX_TOKENS)
        gateway.generate("together", "m", MESSAGES, max_tokens=11,
                         session=session)
        self.assertEqual(session.calls[1]["payload"]["max_tokens"], 11)

    def test_empty_choices_raises_rather_than_indexing(self):
        session = FakeSession([FakeResponse(200, {"choices": []})])
        with self.assertRaises(LLMGatewayError) as caught:
            gateway.generate("together", "m", MESSAGES, session=session)
        self.assertIn("no choices", str(caught.exception))

    def test_missing_content_raises(self):
        session = FakeSession([FakeResponse(200, {"choices": [{"message": {}}]})])
        with self.assertRaises(LLMGatewayError):
            gateway.generate("together", "m", MESSAGES, session=session)


class TestAnthropic(GatewayTestCase):

    def test_happy_path_returns_the_same_shape_as_together(self):
        together = FakeSession([FakeResponse(200, TOGETHER_RESPONSE)])
        anthropic = FakeSession([FakeResponse(200, ANTHROPIC_RESPONSE)])
        a = gateway.generate("together", "m1", MESSAGES, session=together)
        b = gateway.generate("anthropic", "m2", MESSAGES, session=anthropic)
        # The whole point of the gateway: identical keys, identical types.
        self.assertEqual(sorted(a), sorted(b))
        self.assertEqual(a["text"], b["text"])
        self.assertEqual(sorted(a["usage"]), sorted(b["usage"]))

    def test_headers_use_x_api_key_and_a_version(self):
        session = FakeSession([FakeResponse(200, ANTHROPIC_RESPONSE)])
        gateway.generate("anthropic", "claude-haiku-4-5", MESSAGES,
                         session=session)
        headers = session.calls[0]["headers"]
        self.assertIn("x-api-key", headers)
        self.assertEqual(headers["anthropic-version"],
                         constants.ANTHROPIC_VERSION)
        self.assertNotIn("Authorization", headers)

    def test_system_is_hoisted_out_of_the_messages_array(self):
        session = FakeSession([FakeResponse(200, ANTHROPIC_RESPONSE)])
        gateway.generate("anthropic", "m", MESSAGES, session=session)
        payload = session.calls[0]["payload"]
        self.assertEqual(payload["system"], "You are concise.")
        roles = [m["role"] for m in payload["messages"]]
        self.assertNotIn("system", roles)

    def test_both_system_messages_reach_the_payload(self):
        messages = [
            {"role": "system", "content": "You are concise."},
            {"role": "system", "content": "Answer in British English."},
            {"role": "user", "content": "Explain an API."},
        ]
        session = FakeSession([FakeResponse(200, ANTHROPIC_RESPONSE)])
        gateway.generate("anthropic", "m", messages, session=session)
        system = session.calls[0]["payload"]["system"]
        self.assertIn("You are concise.", system)
        self.assertIn("Answer in British English.", system)

    def test_no_system_field_when_there_is_no_system_message(self):
        session = FakeSession([FakeResponse(200, ANTHROPIC_RESPONSE)])
        gateway.generate("anthropic", "m", [{"role": "user", "content": "Hi"}],
                         session=session)
        self.assertNotIn("system", session.calls[0]["payload"])

    def test_the_first_text_block_is_found_past_a_tool_use_block(self):
        # Assuming content[0] is the text would return a tool_use dict here.
        session = FakeSession([FakeResponse(200, ANTHROPIC_TOOL_FIRST)])
        result = gateway.generate("anthropic", "m", MESSAGES, session=session)
        self.assertEqual(result["text"], "The answer is 42.")

    def test_no_text_block_at_all_raises_with_the_block_types(self):
        payload = {"content": [{"type": "tool_use", "name": "lookup"}]}
        session = FakeSession([FakeResponse(200, payload)])
        with self.assertRaises(LLMGatewayError) as caught:
            gateway.generate("anthropic", "m", MESSAGES, session=session)
        self.assertIn("tool_use", str(caught.exception))


class TestUsageNormalization(GatewayTestCase):
    """The providers name token counts differently; callers should not care."""

    def test_together_usage_is_renamed(self):
        session = FakeSession([FakeResponse(200, TOGETHER_RESPONSE)])
        result = gateway.generate("together", "m", MESSAGES, session=session)
        self.assertEqual(result["usage"],
                         {"input_tokens": 18, "output_tokens": 9})

    def test_anthropic_usage_passes_through_the_same_shape(self):
        session = FakeSession([FakeResponse(200, ANTHROPIC_RESPONSE)])
        result = gateway.generate("anthropic", "m", MESSAGES, session=session)
        self.assertEqual(result["usage"],
                         {"input_tokens": 18, "output_tokens": 9})

    def test_missing_usage_does_not_crash(self):
        session = FakeSession([FakeResponse(200, {"choices": [
            {"message": {"content": "hi"}}]})])
        result = gateway.generate("together", "m", MESSAGES, session=session)
        self.assertEqual(result["usage"],
                         {"input_tokens": None, "output_tokens": None})


class TestErrors(GatewayTestCase):

    def test_unsupported_provider_lists_the_supported_ones(self):
        with self.assertRaises(LLMGatewayError) as caught:
            gateway.generate("openai", "m", MESSAGES)
        self.assertIn("together", str(caught.exception))
        self.assertIn("anthropic", str(caught.exception))

    def test_provider_name_is_case_insensitive(self):
        session = FakeSession([FakeResponse(200, TOGETHER_RESPONSE)])
        result = gateway.generate("  TogetherAI ".replace("AI", "").strip(),
                                  "m", MESSAGES, session=session)
        self.assertEqual(result["provider"], "together")

    def test_missing_key_names_the_variable(self):
        os.environ.pop(constants.TOGETHER_KEY_ENV, None)
        with self.assertRaises(LLMGatewayError) as caught:
            gateway.generate("together", "m", MESSAGES)
        self.assertIn(constants.TOGETHER_KEY_ENV, str(caught.exception))

    def test_empty_messages_rejected(self):
        with self.assertRaises(LLMGatewayError):
            gateway.generate("together", "m", [])

    def test_malformed_message_rejected(self):
        with self.assertRaises(LLMGatewayError):
            gateway.generate("together", "m", [{"content": "no role"}])

    def test_http_error_is_raised_not_retried(self):
        session = FakeSession([FakeResponse(401, None, text="bad key")])
        with self.assertRaises(LLMGatewayError) as caught:
            gateway.generate("together", "m", MESSAGES, session=session)
        self.assertIn("401", str(caught.exception))
        self.assertEqual(len(session.calls), 1,
                         "a rejected key should not be retried")

    def test_non_json_body_raises_clearly(self):
        session = FakeSession([FakeResponse(200, None, text="<html>502</html>")])
        with self.assertRaises(LLMGatewayError) as caught:
            gateway.generate("together", "m", MESSAGES, session=session)
        self.assertIn("non-JSON", str(caught.exception))


class TestRetries(GatewayTestCase):

    def test_429_is_retried_and_can_succeed(self):
        session = FakeSession([
            FakeResponse(429, None, text="slow down"),
            FakeResponse(200, TOGETHER_RESPONSE),
        ])
        result = gateway.generate("together", "m", MESSAGES, session=session)
        self.assertEqual(result["text"], "An API is a set of rules.")
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(len(self.slept), 1)

    def test_retry_after_header_is_honoured(self):
        session = FakeSession([
            FakeResponse(429, None, text="slow down", headers={"Retry-After": "7"}),
            FakeResponse(200, TOGETHER_RESPONSE),
        ])
        gateway.generate("together", "m", MESSAGES, session=session)
        self.assertEqual(self.slept, [7.0])

    def test_a_bad_retry_after_falls_back_to_backoff(self):
        session = FakeSession([
            FakeResponse(429, None, text="x", headers={"Retry-After": "soon"}),
            FakeResponse(200, TOGETHER_RESPONSE),
        ])
        gateway.generate("together", "m", MESSAGES, session=session)
        self.assertEqual(self.slept, [constants.INITIAL_BACKOFF])

    def test_backoff_grows(self):
        session = FakeSession([FakeResponse(429, None, text="x")] * constants.MAX_RETRIES)
        with self.assertRaises(LLMGatewayError):
            gateway.generate("together", "m", MESSAGES, session=session)
        self.assertEqual(len(session.calls), constants.MAX_RETRIES)
        self.assertGreater(self.slept[-1], self.slept[0])

    def test_connection_errors_are_retried(self):
        session = FakeSession([
            requests.exceptions.ConnectionError("boom"),
            FakeResponse(200, TOGETHER_RESPONSE),
        ])
        result = gateway.generate("together", "m", MESSAGES, session=session)
        self.assertEqual(result["text"], "An API is a set of rules.")


class TestSecretHandling(GatewayTestCase):
    """An API key must never reach a log, a traceback or a CI transcript."""

    def test_key_is_scrubbed_from_an_http_error_body(self):
        key = os.environ[constants.TOGETHER_KEY_ENV]
        session = FakeSession([
            FakeResponse(400, None, text="invalid key %s supplied" % key)])
        with self.assertRaises(LLMGatewayError) as caught:
            gateway.generate("together", "m", MESSAGES, session=session)
        message = str(caught.exception)
        self.assertNotIn(key, message)
        self.assertIn("[redacted]", message)

    def test_key_is_scrubbed_from_a_connection_error(self):
        key = os.environ[constants.TOGETHER_KEY_ENV]
        session = FakeSession([
            requests.exceptions.ConnectionError("failed with %s" % key)
        ] * constants.MAX_RETRIES)
        with self.assertRaises(LLMGatewayError) as caught:
            gateway.generate("together", "m", MESSAGES, session=session)
        self.assertNotIn(key, str(caught.exception))

    def test_short_strings_are_not_redacted_into_nonsense(self):
        self.assertEqual(gateway._redact("abc", "xy"), "abc")


if __name__ == "__main__":
    unittest.main(verbosity=2)
