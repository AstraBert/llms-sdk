use std::str::FromStr;

use clap::Parser;
use llms_sdk::{ApiType, LLM, LLMRequest, Message, MessagePart, TextPart};

#[derive(Debug, Parser)]
#[command(version = "0.1.0")]
#[command(name = "llms")]
#[command(about, long_about = None)]
/// `llms`: a CLI for quick testing of Anthropic- and OpenAI-compatible LLM APIs.
struct Args {
    /// Type of API: either 'anthropic' or 'openai'
    api: String,
    #[arg(long, short, default_value = None)]
    /// Base URL for the API. If not specified, falls back to the
    /// official Anthropic Messages API and OpenAI Chat
    /// Completions API base URLs.
    base_url: Option<String>,
    #[arg(long, default_value = None)]
    /// API key to use for the request. If not specified, we attempt
    /// to read it from the environment (`ANTHROPIC_API_KEY` for the
    /// Anthropic API, `OPENAI_API_KEY` for the OpenAI API).
    /// If no API keys are found in the environment, we default
    /// to an empty key.
    api_key: Option<String>,
    #[arg(long, short)]
    /// Model to use for the request.
    model: String,
    #[arg(long, short)]
    /// Prompt to send the model
    prompt: String,
    #[arg(long, short, default_value_t = false)]
    /// Whether or not the response should be streamed.
    /// Defaults to false.
    stream: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let api_type = ApiType::from_str(&args.api)?;
    let api_key = if let Some(key) = args.api_key {
        key
    } else {
        match api_type {
            ApiType::OpenAI => {
                if let Ok(k) = std::env::var("OPENAI_API_KEY") {
                    k
                } else {
                    String::new()
                }
            }
            ApiType::Anthropic => {
                if let Ok(k) = std::env::var("ANTHROPIC_API_KEY") {
                    k
                } else {
                    String::new()
                }
            }
        }
    };
    let llm = LLM::default();
    let request = LLMRequest::builder()
        .api_type(api_type)
        .api_key(api_key)
        .model(args.model)
        .stream(args.stream)
        .messages(vec![Message {
            role: llms_sdk::MessageRole::User,
            content: vec![MessagePart::Text(TextPart::new(args.prompt))],
        }])
        .build();
    if !args.stream {
        let response = llm.respond(request).await?;
        for m in response.messages {
            for p in m.content {
                match p {
                    MessagePart::Text(t) => println!("{}", t.text),
                    MessagePart::Thinking(t) => {
                        println!("\x1b[38;5;247mThinking: {}\x1b[0m", t.thinking)
                    }
                    // Should not produce anything apart from text and thinking
                    _ => continue,
                }
            }
        }
    } else {
        eprintln!("Not yet implemented!");
    }
    Ok(())
}
