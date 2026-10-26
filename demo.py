"""Exercise the gateway.

    python3 demo.py --offline     # no keys, no network, no cost
    python3 demo.py               # real calls, needs both API keys

Offline mode is the default thing to run first. It shows the normalized return
shape and demonstrates the system message divergence without spending anything,
which also means it works in CI.
"""

import argparse
import json
import sys

import llm_gateway as gateway
from llm_gateway import LLMGatewayError

MESSAGES = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Explain what an API is in one sentence."},
]

TWO_SYSTEM_MESSAGES = [
    {"role": "system", "content": "You are concise."},
    {"role": "system", "content": "Always answer in British English."},
    {"role": "user", "content": "Explain what an API is."},
]


class CannedSession:
    """Stands in for requests, returning the documented response shapes."""

    TOGETHER = {
        "choices": [{"message": {"content":
                                 "An API is a set of rules that lets programs talk."}}],
        "usage": {"prompt_tokens": 18, "completion_tokens": 11},
    }
    ANTHROPIC = {
        "content": [{"type": "text",
                     "text": "An API is a set of rules that lets programs talk."}],
        "usage": {"input_tokens": 18, "output_tokens": 11},
    }

    def __init__(self, payload):
        self.payload = payload
        self.last = None

    def post(self, url, headers=None, json=None, timeout=None):
        self.last = {"url": url, "headers": headers, "payload": json}
        outer = self

        class Response:
            status_code = 200
            headers = {}
            text = ""

            def json(self):
                return outer.payload
        return Response()


def show_divergence():
    print("The same call, two providers, two different meanings")
    print("=" * 62)
    print("\nMessages sent by the caller:")
    for message in TWO_SYSTEM_MESSAGES:
        print("   %-9s %s" % (message["role"] + ":", message["content"]))

    print("\nTogetherAI keeps system messages in the array, so it receives:")
    for message in TWO_SYSTEM_MESSAGES:
        if message["role"] == "system":
            print("   -", message["content"])

    naive, joined = gateway.system_divergence(TWO_SYSTEM_MESSAGES)
    print("\nAnthropic takes one top-level system field. The obvious loop")
    print("overwrites it each time, so the last message wins:")
    print("   ", repr(naive))
    print("\nThis gateway joins them instead, so Anthropic receives:")
    print("   ", repr(joined))

    lost = [m["content"] for m in TWO_SYSTEM_MESSAGES
            if m["role"] == "system" and m["content"] not in (naive or "")]
    if lost:
        print("\nInstruction the naive version silently drops:")
        for item in lost:
            print("   -", item)
        print("\nTogetherAI would have followed it. Anthropic never saw it.")
        print("That is the gateway failing at the one thing it is for.")


def show_normalized_shape():
    print("\n\nNormalized return shape, both providers")
    print("=" * 62)
    for provider, payload in (("together", CannedSession.TOGETHER),
                              ("anthropic", CannedSession.ANTHROPIC)):
        session = CannedSession(payload)
        import os
        os.environ.setdefault("TOGETHER_API_KEY", "offline-placeholder")
        os.environ.setdefault("ANTHROPIC_API_KEY", "offline-placeholder")
        result = gateway.generate(provider, "demo-model", MESSAGES,
                                  session=session)
        print("\n%s:" % provider)
        print(json.dumps(result, indent=4))
    print("\nIdentical keys, identical types. Calling code never branches on")
    print("which provider answered.")


def live():
    pairs = [("together", "meta-llama/Meta-Llama-3-8B-Instruct-Lite"),
             ("anthropic", "claude-haiku-4-5")]
    failures = 0
    for provider, model in pairs:
        print("\n%s / %s" % (provider, model))
        try:
            result = gateway.generate(provider, model, MESSAGES)
        except LLMGatewayError as problem:
            failures += 1
            print("   failed: %s" % problem)
            continue
        print("   text:  %s" % result["text"].strip()[:160])
        print("   usage: %s" % result["usage"])
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true",
                        help="no keys, no network, no cost")
    args = parser.parse_args(argv)
    if args.offline:
        show_divergence()
        show_normalized_shape()
        return 0
    return live()


if __name__ == "__main__":
    sys.exit(main())
