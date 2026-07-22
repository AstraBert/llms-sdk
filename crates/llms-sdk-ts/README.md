# @cle-does-things/llms-sdk

TypeScript / Node.js bindings for **llms-sdk**, a unified Rust SDK for calling LLM APIs. It exposes the same request/response model for **OpenAI-compatible chat completions** and the **Anthropic Messages API**.

## Features

- Single interface for OpenAI and Anthropic requests.
- Text, image, audio (OpenAI), and document (Anthropic) message parts.
- Structured JSON output via JSON Schema.
- Tool / function calling with provider-specific serialization.
- Streaming responses with text, tool, and reasoning deltas.
- Configurable transient retry policy.
- Native NAPI-RS binary for performance (prebuilt for macOS, Linux, Windows on x64 & arm64).

## Installation

```bash
npm install @cle-does-things/llms-sdk
# or
yarn add @cle-does-things/llms-sdk
```

Prebuilt binaries are included for the most common platforms. If your platform is not covered, the package will attempt to build from source (Rust toolchain required).

## Quick start

```typescript
import { Llm, ApiType, MessageRole } from '@cle-does-things/llms-sdk'

async function main() {
  const request = {
    apiType: ApiType.OpenAI,
    apiKey: process.env.OPENAI_API_KEY!,
    model: 'gpt-5.4-mini',
    messages: [
      {
        role: MessageRole.User,
        content: [{ type: 'Text', field0: { text: 'Hello!' } }],
      },
    ],
    maxOutputTokens: 256,
    temperature: 0.7,
    stream: false,
    parallelToolCalls: false,
  }

  const llm = new Llm()
  const response = await llm.respond(request)
  console.log(response.message.content)
}

main()
```

## Supported API providers

| Provider  | `ApiType` value | Default base URL               |
| --------- | --------------- | ------------------------------ |
| OpenAI    | `'openai'`      | `https://api.openai.com/v1`    |
| Anthropic | `'anthropic'`   | `https://api.anthropic.com/v1` |

## Multimodal input

### Image

```typescript
import { imagePart } from '@cle-does-things/llms-sdk'

const img = imagePart('files/cat.jpeg') // or a Buffer, or a URL
const message = {
  role: MessageRole.User,
  content: [
    { type: 'Text', field0: { text: 'Describe this image.' } },
    { type: 'Image', field0: img },
  ],
}
```

### Audio (OpenAI only)

```typescript
import { audioPart } from '@cle-does-things/llms-sdk'

const aud = audioPart('files/audio.wav') // or a Buffer
const message = {
  role: MessageRole.User,
  content: [
    { type: 'Text', field0: { text: 'Describe this audio.' } },
    { type: 'Audio', field0: aud },
  ],
}
```

### Document (Anthropic only)

```typescript
import { documentPart } from '@cle-does-things/llms-sdk'

const doc = documentPart('files/file.pdf') // or a Buffer, or a URL
const message = {
  role: MessageRole.User,
  content: [
    { type: 'Text', field0: { text: 'Summarize this document.' } },
    { type: 'Document', field0: doc },
  ],
}
```

## Structured output

```typescript
import type { LlmRequest, OutputFormat } from '@cle-does-things/llms-sdk'

const outputFormat: OutputFormat = {
  name: 'capital',
  description: 'Country capital',
  schema: {
    type: 'object',
    properties: {
      country: { type: 'string' },
      capital: { type: 'string' },
    },
    required: ['country', 'capital'],
  },
}

const request: LlmRequest = {
  /* ... */
  outputFormat,
}
```

## Tool use

```typescript
import type { LlmRequest, Tool } from '@cle-does-things/llms-sdk'
import { ToolChoice } from '@cle-does-things/llms-sdk'

const tool: Tool = {
  name: 'get_weather',
  description: 'Return weather for a city.',
  parameters: {
    type: 'object',
    properties: {
      city: { type: 'string' },
    },
    required: ['city'],
  },
}

const request: LlmRequest = {
  /* ... */
  tools: [tool],
  toolChoice: ToolChoice.Auto,
}
```

## Streaming

Set `stream: true` and provide a callback to `streamResponse`:

```typescript
const request = { /* ... */ stream: true }

await llm.streamResponse(request, (err, chunk) => {
  if (err) {
    console.error(err)
    return
  }
  if (!chunk) return

  switch (chunk.type) {
    case 'Delta':
      process.stdout.write(chunk.field0.delta ?? '')
      break
    case 'ToolDelta':
      console.log('Tool delta:', chunk.field0)
      break
    case 'ThinkingDelta':
      console.log('Thinking:', chunk.field0.delta)
      break
    case 'Complete':
      console.log('\nDone:', chunk.field0.message)
      break
  }
})
```

## Retry policy

`Llm` accepts an optional `RetryPolicy`:

```typescript
import { Llm } from '@cle-does-things/llms-sdk'

const llm = new Llm({
  maxRetries: 5,
  minRetryInterval: 500,
  maxRetryInterval: 3000,
  base: 2,
})
```

## TypeScript types

All public types are exported from `index.d.ts`. Key interfaces include:

- `LlmRequest` – request payload
- `LlmResponse` – complete response
- `Message`, `MessagePart` – conversation model
- `ImagePart`, `AudioPart`, `DocumentPart` – multimodal parts
- `Tool`, `ToolCallPart`, `ToolResultPart` – tool use
- `OutputFormat` – structured output schema
- `LLMStreamingResponse` – streaming discriminated union
- `RetryPolicy` – retry configuration

## Tests

Unit tests (no API keys required):

```bash
yarn test
```

Integration tests against live APIs (requires `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`):

```bash
RUN_INTEGRATION_TESTS=true OPENAI_API_KEY=... ANTHROPIC_API_KEY=... yarn test
```

## License

MIT
