use llms_sdk::{
    ApiType, AudioPart, DocumentPart, ImagePart, LLM, LLMRequest, Message, MessagePart,
    MessageRole, ReasoningEffort, RetryPolicy, TextPart, Tool, ToolChoice,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

fn should_run() -> bool {
    std::env::var("RUN_INTEGRATION_TESTS")
        .map(|v| v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn openai_key() -> Option<String> {
    std::env::var("OPENAI_API_KEY").ok()
}

fn anthropic_key() -> Option<String> {
    std::env::var("ANTHROPIC_API_KEY").ok()
}

fn text_msg(text: &str) -> Message {
    Message {
        role: MessageRole::User,
        content: vec![MessagePart::Text(TextPart::new(text))],
    }
}

fn openai_request(model: &str, messages: Vec<Message>) -> LLMRequest {
    LLMRequest {
        api_type: ApiType::OpenAI,
        base_url: Some("https://api.openai.com/v1".to_string()),
        api_key: openai_key().unwrap(),
        model: model.to_string(),
        messages,
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
    }
}

fn anthropic_request(model: &str, messages: Vec<Message>) -> LLMRequest {
    LLMRequest {
        api_type: ApiType::Anthropic,
        base_url: Some("https://api.anthropic.com/v1".to_string()),
        api_key: anthropic_key().unwrap(),
        model: model.to_string(),
        messages,
        max_output_tokens: Some(256),
        temperature: None,
        top_p: None,
        reasoning_effort: Some(ReasoningEffort::Medium),
        prompt_cache_ttl: None,
        stream: false,
        output_format: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: false,
    }
}

#[tokio::test]
#[ignore]
async fn openai_text() {
    if !should_run() || openai_key().is_none() {
        return;
    }
    let req = openai_request("gpt-5.4-mini", vec![text_msg("Say 'hello world' exactly.")]);
    let llm = LLM::new(RetryPolicy::default());
    let resp = llm.respond(req).await.unwrap();
    assert!(!resp.message.content.is_empty());
}

#[tokio::test]
#[ignore]
#[cfg(feature = "fs")]
async fn openai_image() {
    if !should_run() || openai_key().is_none() {
        return;
    }
    let image = ImagePart::try_from_file("files/cat.jpeg".to_string()).unwrap();
    let msg = Message {
        role: MessageRole::User,
        content: vec![
            MessagePart::Text(TextPart::new("Describe this image briefly.")),
            MessagePart::Image(image),
        ],
    };
    let req = openai_request("gpt-5.4-mini", vec![msg]);
    let llm = LLM::new(RetryPolicy::default());
    let resp = llm.respond(req).await.unwrap();
    assert!(!resp.message.content.is_empty());
}

#[tokio::test]
#[ignore]
#[cfg(feature = "fs")]
async fn openai_audio() {
    if !should_run() || openai_key().is_none() {
        return;
    }
    let audio = AudioPart::try_from_file("files/audio.wav".to_string()).unwrap();
    let msg = Message {
        role: MessageRole::User,
        content: vec![
            MessagePart::Text(TextPart::new("Describe this audio briefly.")),
            MessagePart::Audio(audio),
        ],
    };
    let req = openai_request("gpt-audio-1.5", vec![msg]);
    let llm = LLM::new(RetryPolicy::default());
    let resp = llm.respond(req).await.unwrap();
    assert!(!resp.message.content.is_empty());
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Capital {
    country: String,
    capital: String,
}

#[tokio::test]
#[ignore]
async fn openai_structured_output() {
    if !should_run() || openai_key().is_none() {
        return;
    }
    let req = LLMRequest {
        output_format: Some(llms_sdk::OutputFormat {
            name: "capital".to_string(),
            description: "Country capital".to_string(),
            schema: schemars::schema_for!(Capital).into(),
        }),
        ..openai_request(
            "gpt-5.4-mini",
            vec![text_msg("What is the capital of France?")],
        )
    };
    let llm = LLM::new(RetryPolicy::default());
    let resp = llm.respond(req).await.unwrap();
    let text = match resp.message.content.first() {
        Some(MessagePart::Text(t)) => t.text.clone(),
        _ => panic!("expected text"),
    };
    let parsed: Capital = serde_json::from_str(&text).unwrap();
    assert_eq!(parsed.country.to_lowercase(), "france");
    assert!(!parsed.capital.is_empty());
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[tokio::test]
#[ignore]
async fn openai_tool_use() {
    if !should_run() || openai_key().is_none() {
        return;
    }
    let tool = Tool::new::<WeatherArgs>(
        "get_weather",
        "Return weather for a city. Only use this tool.",
    );
    let req = LLMRequest {
        tools: Some(vec![tool]),
        ..openai_request(
            "gpt-5.4-mini",
            vec![text_msg(
                "Call the get_weather tool to tell me what is the weather in Paris",
            )],
        )
    };
    let llm = LLM::new(RetryPolicy::default());
    let resp = llm.respond(req).await.unwrap();
    let has_tool = resp
        .message
        .content
        .iter()
        .any(|p| matches!(p, MessagePart::ToolCall(_)));
    assert!(has_tool);
}

#[tokio::test]
#[ignore]
async fn openai_streaming_text() {
    if !should_run() || openai_key().is_none() {
        return;
    }
    let req = LLMRequest {
        stream: true,
        ..openai_request("gpt-5.4-mini", vec![text_msg("Count to three.")])
    };
    let llm = LLM::new(RetryPolicy::default());
    let mut stream = llm.stream_response(req).await.unwrap();
    let mut saw_delta = false;
    let mut complete = None;
    while let Some(item) = futures_util::StreamExt::next(&mut stream).await {
        match item.unwrap() {
            llms_sdk::LLMStreamingResponse::Delta(_) => saw_delta = true,
            llms_sdk::LLMStreamingResponse::Complete(c) => complete = Some(c),
            _ => {}
        }
    }
    assert!(saw_delta);
    assert!(complete.is_some());
}

#[tokio::test]
#[ignore]
async fn anthropic_text() {
    if !should_run() || anthropic_key().is_none() {
        return;
    }
    let req = anthropic_request(
        "claude-sonnet-5",
        vec![text_msg("Say 'hello world' exactly.")],
    );
    let llm = LLM::new(RetryPolicy::default());
    let resp = llm.respond(req).await.unwrap();
    assert!(!resp.message.content.is_empty());
}

#[tokio::test]
#[ignore]
#[cfg(feature = "fs")]
async fn anthropic_image() {
    if !should_run() || anthropic_key().is_none() {
        return;
    }
    let image = ImagePart::try_from_file("files/cat.jpeg".to_string()).unwrap();
    let msg = Message {
        role: MessageRole::User,
        content: vec![
            MessagePart::Text(TextPart::new("Describe this image briefly.")),
            MessagePart::Image(image),
        ],
    };
    let req = anthropic_request("claude-sonnet-5", vec![msg]);
    let llm = LLM::new(RetryPolicy::default());
    let resp = llm.respond(req).await.unwrap();
    assert!(!resp.message.content.is_empty());
}

#[tokio::test]
#[ignore]
#[cfg(feature = "fs")]
async fn anthropic_document() {
    if !should_run() || anthropic_key().is_none() {
        return;
    }
    let doc = DocumentPart::try_from_pdf_file("files/file.pdf".to_string()).unwrap();
    let msg = Message {
        role: MessageRole::User,
        content: vec![
            MessagePart::Text(TextPart::new("Summarize this document briefly.")),
            MessagePart::Document(doc),
        ],
    };
    let req = anthropic_request("claude-sonnet-5", vec![msg]);
    let llm = LLM::new(RetryPolicy::default());
    let resp = llm.respond(req).await.unwrap();
    assert!(!resp.message.content.is_empty());
}

#[tokio::test]
#[ignore]
async fn anthropic_structured_output() {
    if !should_run() || anthropic_key().is_none() {
        return;
    }
    let req = LLMRequest {
        output_format: Some(llms_sdk::OutputFormat {
            name: "capital".to_string(),
            description: "Country capital".to_string(),
            schema: schemars::schema_for!(Capital).into(),
        }),
        ..anthropic_request(
            "claude-sonnet-5",
            vec![text_msg("What is the capital of France?")],
        )
    };
    let llm = LLM::new(RetryPolicy::default());
    let resp = llm.respond(req).await.unwrap();
    let text = match resp.message.content.first() {
        Some(MessagePart::Text(t)) => t.text.clone(),
        _ => panic!("expected text"),
    };
    let parsed: Capital = serde_json::from_str(&text).unwrap();
    assert_eq!(parsed.country.to_lowercase(), "france");
    assert!(!parsed.capital.is_empty());
}

#[tokio::test]
#[ignore]
async fn anthropic_tool_use() {
    if !should_run() || anthropic_key().is_none() {
        return;
    }
    let tool = Tool::new::<WeatherArgs>(
        "get_weather",
        "Return weather for a city. Only use this tool.",
    );
    let req = LLMRequest {
        tools: Some(vec![tool]),
        tool_choice: Some(ToolChoice::Required),
        ..anthropic_request(
            "claude-sonnet-5",
            vec![text_msg(
                "Call the get_weather tool to tell me what is the weather in Paris",
            )],
        )
    };
    let llm = LLM::new(RetryPolicy::default());
    let resp = llm.respond(req).await.unwrap();
    let has_tool = resp
        .message
        .content
        .iter()
        .any(|p| matches!(p, MessagePart::ToolCall(_)));
    assert!(has_tool);
}

#[tokio::test]
#[ignore]
async fn anthropic_streaming_text() {
    if !should_run() || anthropic_key().is_none() {
        return;
    }
    let req = LLMRequest {
        stream: true,
        ..anthropic_request("claude-sonnet-5", vec![text_msg("Count to three.")])
    };
    let llm = LLM::new(RetryPolicy::default());
    let mut stream = llm.stream_response(req).await.unwrap();
    let mut saw_delta = false;
    let mut complete = None;
    while let Some(item) = futures_util::StreamExt::next(&mut stream).await {
        match item.unwrap() {
            llms_sdk::LLMStreamingResponse::Delta(_) => saw_delta = true,
            llms_sdk::LLMStreamingResponse::Complete(c) => complete = Some(c),
            _ => {}
        }
    }
    assert!(saw_delta);
    assert!(complete.is_some());
}
