# llms-sdk

A unified Rust SDK for calling LLM APIs. It currently supports **OpenAI-compatible chat completions** and the **Anthropic Messages API** through a single request/response model.

> **TypeScript / Node.js bindings are available** in [`crates/llms-sdk-ts`](./crates/llms-sdk-ts).

## Features

- Single interface for OpenAI and Anthropic requests.
- Text, image, audio (OpenAI), and document (Anthropic) message parts.
- Structured JSON output via JSON Schema.
- Tool/function calling with provider-specific serialization.
- Streaming responses with text, tool, and reasoning deltas.
- Configurable transient retry policy.
- Optional CLI binary (`llms`) behind the `cli` feature.
- TypeScript / Node.js bindings via NAPI-RS (`@cle-does-things/llms-sdk`).

## Installation

### Rust

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
llms-sdk = "0.1.1"
```

Enable optional features as needed:

```toml
[dependencies]
llms-sdk = { version = "0.1.1", features = ["cli"] }
```

### TypeScript / Node.js

```bash
npm install @cle-does-things/llms-sdk
# or
yarn add @cle-does-things/llms-sdk
```

See [`crates/llms-sdk-ts/README.md`](./crates/llms-sdk-ts/README.md) for TS examples and API docs.

## Quick start

```rust
use llms_sdk::{ApiType, LLM, LLMRequest, Message, MessagePart, MessageRole, RetryPolicy, TextPart};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request = LLMRequest {
        api_type: ApiType::OpenAI,
        base_url: None,
        api_key: std::env::var("OPENAI_API_KEY")?,
        model: "gpt-5.4-mini".to_string(),
        messages: vec![Message {
            role: MessageRole::User,
            content: vec![MessagePart::Text(TextPart::new("Hello!"))],
        }],
        max_output_tokens: Some(256),
        temperature: Some(0.7),
        top_p: None,
        reasoning_effort: None,
        prompt_cache_ttl: None,
        stream: false,
        output_format: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: false,
    };

    let llm = LLM::new(RetryPolicy::default());
    let response = llm.respond(request).await?;
    println!("{:?}", response);
    Ok(())
}
```

## Building requests

Use `LLMRequest::builder()` to construct requests fluently:

```rust
use llms_sdk::{ApiType, LLMRequest, Message, MessagePart, MessageRole, TextPart};

let request = LLMRequest::builder()
    .api_type(ApiType::Anthropic)
    .api_key(std::env::var("ANTHROPIC_API_KEY").unwrap())
    .model("claude-5-sonnet".to_string())
    .messages(vec![Message {
        role: MessageRole::User,
        content: vec![MessagePart::Text(TextPart::new("Hi!"))],
    }])
    .max_output_tokens(256)
    .build();
```

## Multimodal input

### Image

```rust
use llms_sdk::{ImagePart, Message, MessagePart, MessageRole, TextPart};

let image = ImagePart::try_from_file("files/cat.jpeg".to_string())?;
let message = Message {
    role: MessageRole::User,
    content: vec![
        MessagePart::Text(TextPart::new("Describe this image.")),
        MessagePart::Image(image),
    ],
};
```

### Audio (OpenAI only)

```rust
use llms_sdk::{AudioPart, Message, MessagePart, MessageRole, TextPart};

let audio = AudioPart::try_from_file("files/audio.wav".to_string())?;
let message = Message {
    role: MessageRole::User,
    content: vec![
        MessagePart::Text(TextPart::new("Describe this audio.")),
        MessagePart::Audio(audio),
    ],
};
```

### Document (Anthropic only)

```rust
use llms_sdk::{DocumentPart, Message, MessagePart, MessageRole, TextPart};

let doc = DocumentPart::try_from_pdf_file("files/file.pdf".to_string())?;
let message = Message {
    role: MessageRole::User,
    content: vec![
        MessagePart::Text(TextPart::new("Summarize this document.")),
        MessagePart::Document(doc),
    ],
};
```

## Structured output

```rust
use llms_sdk::{LLMRequest, OutputFormat};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct Capital {
    country: String,
    capital: String,
}

let request = LLMRequest {
    output_format: Some(OutputFormat {
        name: "capital".to_string(),
        description: "Country capital".to_string(),
        schema: schemars::schema_for!(Capital).into(),
    }),
    ..request
};
```

## Tool use

```rust
use llms_sdk::{Tool, ToolChoice};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

let tool = Tool::new::<WeatherArgs>("get_weather", "Return weather for a city.");
let request = LLMRequest {
    tools: Some(vec![tool]),
    tool_choice: Some(ToolChoice::Auto),
    ..request
};
```

## Streaming

Set `stream: true` and consume the returned stream:

```rust
use futures_util::StreamExt;
use llms_sdk::LLMStreamingResponse;

let request = LLMRequest { stream: true, ..request };
let mut stream = llm.stream_response(request).await?;

while let Some(item) = stream.next().await {
    match item? {
        LLMStreamingResponse::Delta(d) => println!("{}", d.delta.unwrap_or_default()),
        LLMStreamingResponse::Complete(c) => println!("done: {:?}", c),
        _ => {}
    }
}
```

## CLI

Install and run the CLI with the `cli` feature enabled:

```bash
cargo install llms-sdk --features cli
llms --help
```

> _The CLI is mostly meant for testing purposes_

## Tests

Run unit tests:

```bash
cargo test --all-features
```

Run integration tests against live APIs (requires `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`):

```bash
RUN_INTEGRATION_TESTS=true OPENAI_API_KEY=... ANTHROPIC_API_KEY=... cargo test --all-features --test integration_test -- --ignored
```

For more details, refer to [CONTRIBUTING.md](./CONTRIBUTING.md)

## License

This project is licensed under the MIT License.
