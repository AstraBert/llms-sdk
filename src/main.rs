use std::{fs, str::FromStr};

use clap::Parser;
use futures_util::StreamExt;
use llms_sdk::{
    ApiType, LLM, LLMRequest, Message, MessagePart, OutputFormat, ReasoningEffort, TextPart, Tool,
};

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
    #[arg(long, short, default_value = None)]
    /// Effort the model should put into
    /// reasoning while responding to
    /// the prompt.
    /// Allowed values: 'none' (default),
    /// 'minimal', 'low', 'medium', 'high',
    /// 'maximum'
    reasoning_effort: Option<String>,
    #[arg(long, default_value = None)]
    /// One or more JSON files containing tool
    /// definitions that follow the shape required
    /// by the Tool struct.
    tool_file: Option<Vec<String>>,
    #[arg(long, default_value = None)]
    /// A JSON file containing the output
    /// schema for the response.
    json_schema_file: Option<String>,
}

fn tool_from_file(fl: String) -> Result<Tool, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(fl)?;
    let tool: Tool = serde_json::from_str(&content)?;
    Ok(tool)
}

fn schema_from_file(fl: String) -> Result<OutputFormat, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(fl)?;
    let schema: OutputFormat = serde_json::from_str(&content)?;
    Ok(schema)
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
    let reasoning = match args.reasoning_effort {
        Some(r) => Some(ReasoningEffort::from_str(&r)?),
        None => None,
    };
    let llm = LLM::default();
    let tools = if let Some(tfs) = args.tool_file {
        let mut ts = vec![];
        for tf in tfs {
            ts.push(tool_from_file(tf)?);
        }
        Some(ts)
    } else {
        None
    };
    let mut request = LLMRequest::builder()
        .api_type(api_type)
        .api_key(api_key)
        .model(args.model)
        .stream(args.stream)
        .messages(vec![Message {
            role: llms_sdk::MessageRole::User,
            content: vec![MessagePart::Text(TextPart::new(args.prompt))],
        }])
        .build();
    if let Some(mut t) = tools {
        request.tools.get_or_insert_with(Vec::new).append(&mut t);
    }
    if let Some(reason) = reasoning {
        request.reasoning_effort = Some(reason);
    }
    if let Some(fl) = args.json_schema_file {
        let schema = schema_from_file(fl)?;
        request.output_format = Some(schema);
    }
    if !args.stream {
        let response = llm.respond(request).await?;
        for p in response.message.content {
            match p {
                MessagePart::Text(t) => println!("{}", t.text),
                MessagePart::Thinking(t) => {
                    println!("\x1b[38;5;247mThinking: {}\x1b[0m", t.thinking)
                }
                // Should not produce anything apart from text and thinking
                _ => continue,
            }
        }
    } else {
        let mut stream = llm.stream_response(request).await?;
        while let Some(event) = stream.next().await {
            let event = event.map_err(|e| e.to_string())?;
            match event {
                llms_sdk::LLMStreamingResponse::Delta(d) => {
                    print!("{}", d.delta.unwrap_or_default())
                }
                llms_sdk::LLMStreamingResponse::ToolDelta(tc) => {
                    print!("{}", tc.partial_arguments)
                }
                llms_sdk::LLMStreamingResponse::Complete(c) => println!(
                    "\n\n\x1b[38;5;247mInput Tokens: {:?}; Output Tokens: {:?}; Tool Calls: {:?}",
                    c.usage.map_or(0, |r| r.input_tokens),
                    c.usage.map_or(0, |r| r.output_tokens),
                    c.tool_calls.map_or(0, |r| r.len())
                ),
            }
        }
    }
    Ok(())
}
