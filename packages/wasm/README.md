# @cle-does-things/llms-sdk-wasm

WASM bindings for `llms-sdk`, to call OpenAI- and Anthropic-compatible LLMs from the browser.

## Installation

```bash
npm install @cle-does-things/llms-sdk-wasm
```

Or use directly in a browser via a bundler that supports WASM (e.g. Vite, Webpack, Rollup).

## Quick Start

```javascript
import init, { chat } from "@cle-does-things/llms-sdk-wasm";

await init();

const response = await chat({
  api_type: "openai",
  api_key: "sk-...",
  model: "gpt-5.4-mini",
  messages: [
    { role: "user", content: [{ type: "text", text: "Hello!" }] }
  ],
  stream: false,
  parallel_tool_calls: false,
});

console.log(response.message.content);
```

## API

### Initialization

```javascript
import init from "@cle-does-things/llms-sdk-wasm";
await init(); // loads and instantiates the WASM module
```

### `chat(request, retry_policy?)`

Sends a single non-streaming chat request and returns the full response.

**Parameters:**
- `request: LLMRequest` — the chat request (see [LLMRequest](#llmrequest))

**Returns:** `Promise<LLMResponse>`

```javascript
const response = await chat({
  api_type: "openai",
  base_url: "https://api.openai.com/v1",  // optional
  api_key: "sk-...",
  model: "gpt-5.4-mini",
  messages: [
    { role: "system", content: [{ type: "text", text: "You are a helpful assistant." }] },
    { role: "user", content: [{ type: "text", text: "Explain quantum computing." }] }
  ],
  max_output_tokens: 256,
  temperature: 0.7,
  top_p: 1.0,
  stream: false,
  parallel_tool_calls: false,
});

// response.id          — provider-generated response ID
// response.created_at  — unix timestamp
// response.message     — { role, content: MessagePart[] }
// response.usage       — { input_tokens, output_tokens, cache_read_tokens, cache_write_tokens }
```

### `streamChat(request, callback, retry_policy?)`

Streams a chat response, invoking the callback for each chunk.

**Parameters:**
- `request: LLMRequest` — the chat request
- `callback: (error: any, chunk: LLMStreamingResponse) => void` — called on every stream event

**Returns:** `Promise<void>`

```javascript
await streamChat(request, (err, chunk) => {
  if (err) {
    console.error("Stream error:", err);
    return;
  }

  switch (chunk.type) {
    case "delta":
      console.log("Text:", chunk.delta);
      console.log("Done?", chunk.stop);
      break;
    case "thinkingDelta":
      console.log("Reasoning:", chunk.delta);
      break;
    case "toolDelta":
      console.log("Tool call:", chunk.name, chunk.partial_arguments);
      break;
    case "complete":
      console.log("Final message:", chunk.message);
      console.log("Usage:", chunk.usage);
      break;
  }
});
```

### Content Helpers

Build `MessagePart` objects for multimodal inputs:

```javascript
import { imagePart, documentPart, audioPart } from "@cle-does-things/llms-sdk-wasm";

// Image from URL or Uint8Array
const img = imagePart("https://example.com/image.png");
const imgFromBytes = imagePart({ bytes: new Uint8Array([...]) });

// Document (PDF, etc.) from URL or Uint8Array
const doc = documentPart("https://example.com/doc.pdf");

// Audio from Uint8Array
const audio = audioPart({ bytes: new Uint8Array([...]) });
```

## Types

### `LLMRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `api_type` | `"openai" \| "anthropic"` | ✅ | Provider type |
| `base_url` | `string` | | Override the default API base URL |
| `api_key` | `string` | ✅ | API key |
| `model` | `string` | ✅ | Model identifier |
| `messages` | `Message[]` | ✅ | Conversation history |
| `max_output_tokens` | `number` | | Maximum tokens to generate |
| `temperature` | `number` | | Sampling temperature |
| `top_p` | `number` | | Nucleus sampling |
| `reasoning_effort` | `ReasoningEffort` | | Control reasoning depth (`none` … `maximum`) |
| `prompt_cache_ttl` | `string` | | Prompt cache TTL hint |
| `stream` | `boolean` | ✅ | Enable streaming |
| `output_format` | `OutputFormat` | | Structured output JSON schema |
| `tools` | `Tool[]` | | Available function tools |
| `tool_choice` | `ToolChoice` | | Tool selection mode (`auto`, `none`, `required`) |
| `parallel_tool_calls` | `boolean` | ✅ | Allow parallel tool calls |

### `Message`

```typescript
interface Message {
  role: "user" | "assistant" | "tool" | "system";
  content: MessagePart[];
}
```

### `MessagePart`

A discriminated union of content types:

| Tag | Fields |
|-----|--------|
| `{ type: "text" }` | `text: string` |
| `{ type: "image" }` | `image_data: string`, `is_base64: boolean`, `mime_type?: string` |
| `{ type: "document" }` | `document_data: string`, `is_base64: boolean`, `mime_type?: string` |
| `{ type: "audio" }` | `audio_data: string`, `mime_type: string` |
| `{ type: "toolCall" }` | `id: string`, `name: string`, `arguments: string` |
| `{ type: "toolResult" }` | `tool_call_id: string`, `result: string` |
| `{ type: "thinking" }` | `thinking: string`, `signature?: string` |

### `LLMResponse`

```typescript
interface LLMResponse {
  id: string;
  created_at?: number;
  message: Message;
  usage: LLMUsage;
}
```

### `LLMUsage`

```typescript
interface LLMUsage {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens?: number;
  cache_write_tokens?: number;
  other_tokens?: Map<string, number>;
}
```

### Streaming Types

```typescript
type LLMStreamingResponse =
  | { type: "delta"; response_id: string; created_at?: number; delta?: string; stop: boolean }
  | { type: "thinkingDelta"; response_id: string; created_at?: number; delta?: string }
  | { type: "toolDelta"; response_id: string; tool_call_id: string; name: string; partial_arguments: string }
  | { type: "complete"; id: string; created_at?: number; message: Message; deltas: LLMStreamingDelta[]; thinking_deltas?: LLMThinkingDelta[]; usage?: LLMUsage; tool_calls?: ToolCallPart[] };
```

## Demo

A complete browser demo is included in `demo.html`. Open it in a browser (via a local server) after building the WASM package:

```bash
cd packages/wasm
wasm-pack build --target web
# serve the directory, e.g.
npx serve .
```

Then open `http://localhost:3000/demo.html`.

## Building from Source

Requires [Rust](https://rustup.rs/) and [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/):

```bash
npm run build
```

The compiled package is output to `pkg/`.

## License

MIT
