# Multi-Provider LLM Gateway

Dataquest guided project, *Build a Multi-Provider LLM Gateway*. One
`generate()` function that talks to TogetherAI or Anthropic and returns the
same shape either way, built on `requests` rather than each provider's SDK so
the HTTP mechanics stay visible.

```python
from llm_gateway import generate

messages = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Explain what an API is in one sentence."},
]

generate(provider="together",  model="meta-llama/Meta-Llama-3-8B-Instruct-Lite", messages=messages)
generate(provider="anthropic", model="claude-haiku-4-5", messages=messages)
```

Both return:

```python
{"provider": ..., "model": ..., "text": ..., "usage": {"input_tokens": ..., "output_tokens": ...}}
```

## Running it

```
python3 demo.py --offline              # no keys, no network, no cost
python3 -m unittest -v test_llm_gateway   # 37 tests, also no keys
```

For real calls, export both keys (`.env.example` lists them) and run
`python3 demo.py`.

## The bug this is built around

The brief mentions in passing that if several system messages arrive, the
Anthropic loop "will keep overwriting `system_text` with each new one, so the
last system message wins", and calls it an edge case worth knowing about.

It is worse than an edge case, because of where it sits. TogetherAI takes
system messages inside the `messages` array, so it receives **all** of them.
Anthropic takes a single top-level `system` field, so the overwriting loop
sends **one**. Same gateway, same call, two different sets of instructions:

```
Messages sent by the caller:
   system:   You are concise.
   system:   Always answer in British English.
   user:     Explain what an API is.

TogetherAI keeps system messages in the array, so it receives:
   - You are concise.
   - Always answer in British English.

Anthropic takes one top-level system field. The obvious loop
overwrites it each time, so the last message wins:
    'Always answer in British English.'

Instruction the naive version silently drops:
   - You are concise.
```

A gateway exists so that calling code does not have to care which provider
answered. A gateway that quietly changes the prompt depending on the provider
has given up the only thing it was for, and it does so without an error, a
warning, or any difference in the return shape. You would notice it as
"Anthropic ignores my instructions sometimes".

`split_system_messages` joins every system message with a blank line, so both
providers receive the same instruction set. `system_divergence()` returns the
naive result alongside the joined one, which is what `demo.py --offline`
prints and what `test_the_naive_loop_drops_an_instruction` pins down.

Joining is a choice, not the only answer. Raising on multiple system messages
would also be defensible, and arguably safer for a library. Joining wins here
because it matches what TogetherAI already does with the same input, and
matching the other provider is the whole job.

## Why the tests do not call the providers

The brief's suggested test is to call both providers and check that text comes
back. That confirms the keys work and little else, because **both providers
returning some text looks exactly like both providers returning the right
text.** A normalization bug produces two perfectly healthy responses.

So the provider contracts are pinned as fixtures taken from the documented
response shapes, and the suite asserts on the request the gateway *builds* and
the result it *returns*:

- TogetherAI gets `Authorization: Bearer` and no `x-api-key`; Anthropic gets
  `x-api-key` plus `anthropic-version`, and no `Authorization`.
- TogetherAI keeps `system` in the array; Anthropic hoists it to a top-level
  field and leaves no system role in `messages`.
- Both results have identical keys and identical usage keys. That one assertion
  is the gateway's contract, and it is the one a live test cannot make.

37 tests, no network, no keys, no cost, so they run in CI on every commit.

## The other things it handles

**Anthropic may not lead with text.** The brief flags that `content[0]` is not
guaranteed to be a text block, since `tool_use` and others exist. The parser
takes the first block whose `type` is `text`, and raises with the actual block
types if there is none. Tested with a `tool_use` block in front.

**429 is retried, nothing else is.** A rate limit means the request was fine
and merely early, so it backs off exponentially and honours `Retry-After`
(falling back to the backoff if that header is unparseable). A 401 is raised
immediately after a single attempt, because retrying a rejected key only
wastes time. There is a test asserting exactly that call count.

**Token usage is normalized.** TogetherAI reports `prompt_tokens` and
`completion_tokens`; Anthropic reports `input_tokens` and `output_tokens`. Left
alone, a caller that wanted to log or bill on tokens would still need
provider-specific code, which is the thing the gateway removes. Both are mapped
onto `input_tokens` and `output_tokens`.

**Keys never reach a log.** Providers sometimes echo part of a request back in
an error body, and error strings end up in CI output, which is durable and
often public. Every error message passes through `_redact()`, and two tests
assert the key does not survive into the exception text.

**Responses are read with `.get()` throughout.** A provider response is not
guaranteed to match its own documentation, and a `KeyError` three levels deep
tells the caller nothing. Missing fields raise `LLMGatewayError` naming what
was actually present.

## Files

| File | Contents |
|---|---|
| `constants.py` | endpoints, API version, defaults, retry settings |
| `llm_gateway.py` | `generate()`, the two provider functions, redaction, retries |
| `test_llm_gateway.py` | 37 tests against pinned provider contracts |
| `demo.py` | `--offline` for the divergence and the shape, live for real calls |
| `.env.example` | the two environment variables to set |

## What I would add next

Streaming, which both providers support and which the normalized return shape
would need to change for: a `text` string becomes an iterator, and the usage
figures only arrive at the end. That is a bigger change than it looks, and it
is the reason real gateways expose `generate()` and `stream()` as separate
functions rather than adding a flag.
