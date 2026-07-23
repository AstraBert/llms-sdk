use std::collections::BTreeMap;

use async_stream::stream;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use reqwest_middleware::ClientBuilder;
use reqwest_retry::{RetryTransientMiddleware, policies::ExponentialBackoff};
use schemars::Schema;
use serde::{Deserialize, Serialize};

use crate::{
    ApiType, AudioPart, CHAT_COMPLETIONS_ENDPOINT, ImagePart, LLMRequest, LLMResponse, LLMStream,
    LLMStreamingComplete, LLMStreamingDelta, LLMStreamingResponse, LLMToolDelta, LLMUsage, Message,
    MessagePart, MessageRole, OutputFormat, ReasoningEffort, RetryPolicy, TextPart, ToolCallPart,
    ToolChoice, ToolResultPart,
    errors::{StreamParamError, UnsupportedPartType},
    is_valid_json,
};

/// Client for sending requests to OpenAI-compatible chat completion APIs.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAIClient {
    /// Retry policy applied to API requests made by this client.
    pub retry_policy: RetryPolicy,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIMessageRole {
    Developer,
    User,
    Assistant,
    Tool,
}

impl From<MessageRole> for OpenAIMessageRole {
    fn from(value: MessageRole) -> Self {
        match value {
            MessageRole::Assistant => Self::Assistant,
            MessageRole::System => Self::Developer,
            MessageRole::Tool => Self::Tool,
            MessageRole::User => Self::User,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAITextPart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub text: String,
}

impl From<TextPart> for OpenAITextPart {
    fn from(value: TextPart) -> Self {
        Self {
            part_type: "text".to_string(),
            text: value.text,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIImageUrl {
    pub url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIImagePart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub image_url: OpenAIImageUrl,
}

impl From<ImagePart> for OpenAIImagePart {
    fn from(value: ImagePart) -> Self {
        let image_url = if value.is_base64 {
            format!(
                "data:{};base64,{}",
                value.mime_type.as_ref().unwrap(),
                value.data
            )
        } else {
            value.data.clone()
        };
        Self {
            part_type: "image_url".to_string(),
            image_url: OpenAIImageUrl { url: image_url },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIInputAudio {
    data: String,
    format: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIAudioPart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub input_audio: OpenAIInputAudio,
}

impl From<AudioPart> for OpenAIAudioPart {
    fn from(value: AudioPart) -> Self {
        Self {
            part_type: "input_audio".to_string(),
            input_audio: OpenAIInputAudio {
                data: value.data,
                format: {
                    match value.mime_type.as_str() {
                        "audio/mpeg" | "audio/mp3" => "mp3".to_string(),
                        "audio/wav" | "audio/vnd.wav" | "audio/vnd.wave" => "wav".to_string(),
                        _ => unreachable!("This branch should not be reached"),
                    }
                },
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCallPart {
    pub id: String,
    pub function: OpenAIFunctionCall,
    #[serde(rename = "type")]
    pub part_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolResultPart {
    tool_call_id: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIMessagePart {
    Text(OpenAITextPart),
    Audio(OpenAIAudioPart),
    Image(OpenAIImagePart),
    ToolCall(OpenAIToolCallPart),
    ToolResult(OpenAIToolResultPart),
}

impl TryFrom<MessagePart> for OpenAIMessagePart {
    type Error = UnsupportedPartType;
    #[allow(unreachable_patterns)]
    fn try_from(value: MessagePart) -> Result<Self, Self::Error> {
        match value {
            MessagePart::Text(t) => Ok(Self::Text(OpenAITextPart::from(t))),
            MessagePart::Audio(a) => Ok(Self::Audio(OpenAIAudioPart::from(a))),
            MessagePart::Image(i) => Ok(Self::Image(OpenAIImagePart::from(i))),
            MessagePart::Thinking(_) => Err(UnsupportedPartType {
                part_type: "thinking".to_string(),
                api_type: ApiType::OpenAI.to_string(),
            }),
            MessagePart::ToolCall(tc) => Ok(Self::ToolCall(OpenAIToolCallPart {
                id: tc.id,
                function: OpenAIFunctionCall {
                    name: tc.name,
                    arguments: tc.arguments,
                },
                part_type: "function".to_string(),
            })),
            MessagePart::ToolResult(tr) => Ok(Self::ToolResult(OpenAIToolResultPart {
                tool_call_id: tr.tool_call_id,
                content: tr.result,
            })),
            MessagePart::Document(_) => Err(UnsupportedPartType {
                part_type: "document".to_string(),
                api_type: ApiType::OpenAI.to_string(),
            }),
            _ => Err(UnsupportedPartType {
                part_type: "unknown".to_string(),
                api_type: ApiType::OpenAI.to_string(),
            }),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAISimpleMessage {
    pub role: OpenAIMessageRole,
    pub content: Vec<OpenAIMessagePart>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIAssistantMessage {
    pub role: OpenAIMessageRole,
    pub content: Vec<OpenAIMessagePart>,
    pub tool_calls: Option<Vec<OpenAIToolCallPart>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIToolResultMessage {
    pub role: OpenAIMessageRole,
    pub content: String,
    pub tool_call_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum OpenAIMessage {
    User(OpenAISimpleMessage),
    Developer(OpenAISimpleMessage),
    Assistant(OpenAIAssistantMessage),
    Tool(OpenAIToolResultMessage),
}

impl TryFrom<Message> for OpenAIMessage {
    type Error = UnsupportedPartType;

    fn try_from(value: Message) -> Result<Self, Self::Error> {
        match value.role {
            MessageRole::User => {
                let mut content = vec![];
                for p in value.content {
                    let part = OpenAIMessagePart::try_from(p)?;
                    match part {
                        OpenAIMessagePart::ToolCall(_) | OpenAIMessagePart::ToolResult(_) => {
                            return Err(UnsupportedPartType {
                                part_type: "tool_call or tool_result (for user/system message)"
                                    .to_string(),
                                api_type: ApiType::OpenAI.to_string(),
                            });
                        }
                        _ => content.push(part),
                    }
                }
                Ok(Self::User(OpenAISimpleMessage {
                    role: OpenAIMessageRole::User,
                    content,
                }))
            }
            MessageRole::System => {
                let mut content = vec![];
                for p in value.content {
                    let part = OpenAIMessagePart::try_from(p)?;
                    match part {
                        OpenAIMessagePart::ToolCall(_) | OpenAIMessagePart::ToolResult(_) => {
                            return Err(UnsupportedPartType {
                                part_type: "tool_call or tool_result (for user/system message)"
                                    .to_string(),
                                api_type: ApiType::OpenAI.to_string(),
                            });
                        }
                        _ => content.push(part),
                    }
                }
                Ok(Self::Developer(OpenAISimpleMessage {
                    role: OpenAIMessageRole::Developer,
                    content,
                }))
            }
            MessageRole::Assistant => {
                let mut content = vec![];
                let mut tool_calls: Option<Vec<OpenAIToolCallPart>> = None;
                for p in value.content {
                    let part = OpenAIMessagePart::try_from(p)?;
                    match part {
                        OpenAIMessagePart::ToolCall(tc) => {
                            tool_calls.get_or_insert_with(Vec::new).push(tc)
                        }
                        OpenAIMessagePart::ToolResult(_) => {
                            return Err(UnsupportedPartType {
                                part_type: "tool_result (for assistant message)".to_string(),
                                api_type: ApiType::OpenAI.to_string(),
                            });
                        }
                        _ => content.push(part),
                    }
                }
                Ok(Self::Assistant(OpenAIAssistantMessage {
                    role: OpenAIMessageRole::Assistant,
                    content,
                    tool_calls,
                }))
            }
            MessageRole::Tool => {
                if value.content.len() != 1 {
                    return Err(UnsupportedPartType {
                        part_type: "multiple tool results".to_string(),
                        api_type: ApiType::OpenAI.to_string(),
                    });
                }
                let mut content = String::new();
                let mut tool_call_id: Option<String> = None;
                for p in value.content {
                    let part = OpenAIMessagePart::try_from(p)?;
                    match part {
                        OpenAIMessagePart::ToolResult(t) => {
                            content += &t.content;
                            tool_call_id = Some(t.tool_call_id);
                        }
                        _ => {
                            return Err(UnsupportedPartType {
                                part_type:
                                    "non-tool_result parts are not supported as tool results"
                                        .to_string(),
                                api_type: ApiType::OpenAI.to_string(),
                            });
                        }
                    }
                }
                if let Some(tid) = tool_call_id {
                    Ok(Self::Tool(OpenAIToolResultMessage {
                        role: OpenAIMessageRole::Tool,
                        content,
                        tool_call_id: tid,
                    }))
                } else {
                    Err(UnsupportedPartType {
                        part_type: "tool_result does not have a tool_call_id".to_string(),
                        api_type: ApiType::OpenAI.to_string(),
                    })
                }
            }
        }
    }
}

impl From<OpenAIMessage> for Message {
    fn from(val: OpenAIMessage) -> Self {
        match val {
            OpenAIMessage::User(u) => {
                let mut content = vec![];
                for c in u.content {
                    match c {
                        OpenAIMessagePart::Text(t) => {
                            content.push(MessagePart::Text(TextPart { text: t.text }))
                        }
                        OpenAIMessagePart::Audio(a) => {
                            content.push(MessagePart::Audio(AudioPart {
                                data: a.input_audio.data,
                                mime_type: format!("audio/{}", a.input_audio.format),
                            }))
                        }
                        OpenAIMessagePart::Image(i) => {
                            let (is_base64, mime_type, data) =
                                if i.image_url.url.starts_with("data:") {
                                    let (split, data) = i
                                        .image_url
                                        .url
                                        .split_once(";")
                                        .expect("Should return a clean split around ';'");
                                    let format = split.replace("data:", "").trim().to_owned();
                                    (
                                        true,
                                        Some(format),
                                        data.replacen("base64,", "", 1).to_owned(),
                                    )
                                } else {
                                    (false, None, i.image_url.url)
                                };
                            content.push(MessagePart::Image(ImagePart {
                                data,
                                is_base64,
                                mime_type,
                            }));
                        }
                        // user is not supposed to have tool calls or results
                        _ => continue,
                    }
                }
                Message {
                    role: MessageRole::User,
                    content,
                }
            }
            OpenAIMessage::Assistant(a) => {
                let mut content = vec![];
                for c in a.content {
                    match c {
                        OpenAIMessagePart::Text(t) => {
                            content.push(MessagePart::Text(TextPart { text: t.text }))
                        }
                        // assitant is not supposed to produce anything other than text, tool calls are below
                        _ => continue,
                    }
                }
                if let Some(tcs) = a.tool_calls {
                    for tc in tcs {
                        content.push(MessagePart::ToolCall(ToolCallPart {
                            id: tc.id,
                            name: tc.function.name,
                            arguments: tc.function.arguments,
                        }));
                    }
                }
                Message {
                    role: MessageRole::Assistant,
                    content,
                }
            }
            OpenAIMessage::Developer(d) => {
                let mut content = vec![];
                for c in d.content {
                    match c {
                        OpenAIMessagePart::Text(t) => {
                            content.push(MessagePart::Text(TextPart { text: t.text }))
                        }
                        // system messages should only contain text
                        _ => continue,
                    }
                }
                Message {
                    role: MessageRole::System,
                    content,
                }
            }
            OpenAIMessage::Tool(t) => Message {
                role: MessageRole::Tool,
                content: vec![MessagePart::ToolResult(ToolResultPart {
                    tool_call_id: t.tool_call_id,
                    result: t.content,
                })],
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum PromptCacheMode {
    #[default]
    Implicit,
    Explicit,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PromptCacheOptions {
    pub mode: PromptCacheMode,
    pub ttl: String,
}

impl Default for PromptCacheOptions {
    fn default() -> Self {
        Self {
            mode: PromptCacheMode::default(),
            ttl: "30m".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl From<ReasoningEffort> for OpenAIReasoningEffort {
    fn from(value: ReasoningEffort) -> Self {
        match value {
            ReasoningEffort::None => Self::None,
            ReasoningEffort::Minimal => Self::Minimal,
            ReasoningEffort::Low => Self::Low,
            ReasoningEffort::High => Self::High,
            ReasoningEffort::Medium => Self::Medium,
            ReasoningEffort::Maximum => Self::Max,
            ReasoningEffort::Xhigh => Self::Xhigh,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ResponseFormatType {
    #[default]
    JsonSchema,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: ResponseFormatType,
    json_schema: OutputFormat,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIStreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIFunction {
    pub name: String,
    pub description: String,
    pub parameters: Schema,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIFunction,
}

/// Serialized request body sent to the OpenAI chat completions endpoint.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_options: Option<PromptCacheOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<OpenAIReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OpenAIStreamOptions>,
}

impl TryFrom<LLMRequest> for OpenAIRequest {
    type Error = UnsupportedPartType;
    fn try_from(value: LLMRequest) -> Result<Self, Self::Error> {
        let mut messages = vec![];
        for m in value.messages {
            messages.push(OpenAIMessage::try_from(m)?);
        }
        let stream_options = if value.stream {
            Some(OpenAIStreamOptions {
                include_usage: true,
            })
        } else {
            None
        };
        let mut tools = None;
        let mut parallel_tool_calls = None;
        if let Some(ts) = value.tools
            && !ts.is_empty()
        {
            parallel_tool_calls = Some(value.parallel_tool_calls);
            for t in ts {
                tools.get_or_insert_with(Vec::new).push(OpenAITool {
                    tool_type: "function".to_string(),
                    function: OpenAIFunction {
                        name: t.name,
                        description: t.description,
                        parameters: t.parameters,
                    },
                });
            }
        }
        Ok(Self {
            model: value.model,
            messages,
            max_completion_tokens: value.max_output_tokens,
            temperature: value.temperature,
            tool_choice: value.tool_choice,
            tools,
            prompt_cache_options: value.prompt_cache_ttl.map(|ttl| PromptCacheOptions {
                mode: PromptCacheMode::default(),
                ttl,
            }),
            parallel_tool_calls,
            stream: value.stream,
            top_p: value.top_p,
            response_format: value.output_format.map(|f| ResponseFormat {
                format_type: ResponseFormatType::default(),
                json_schema: f,
            }),
            reasoning_effort: value.reasoning_effort.map(OpenAIReasoningEffort::from),
            stream_options,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct OpenAIPromptTokensDetails {
    pub cached_tokens: u32,
    pub audio_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct OpenAICompletionTokensDetails {
    pub reasoning_tokens: u32,
    pub audio_tokens: u32,
    pub accepted_prediction_tokens: u32,
    pub rejected_prediction_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub prompt_tokens_details: OpenAIPromptTokensDetails,
    pub completion_tokens_details: OpenAICompletionTokensDetails,
}

impl From<OpenAIUsage> for LLMUsage {
    fn from(value: OpenAIUsage) -> Self {
        Self {
            input_tokens: value.prompt_tokens,
            output_tokens: value.completion_tokens,
            cache_read_tokens: Some(value.prompt_tokens_details.cached_tokens),
            cache_write_tokens: None,
            other_tokens: Some(
                value.prompt_tokens_details.audio_tokens
                    + value.completion_tokens_details.audio_tokens
                    + value.completion_tokens_details.reasoning_tokens,
            ),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAISimplifiedUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

impl From<OpenAISimplifiedUsage> for LLMUsage {
    fn from(value: OpenAISimplifiedUsage) -> Self {
        Self {
            input_tokens: value.prompt_tokens,
            output_tokens: value.completion_tokens,
            cache_read_tokens: None,
            cache_write_tokens: None,
            other_tokens: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompletionMessage {
    pub role: OpenAIMessageRole,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCallPart>>,
}

impl From<CompletionMessage> for Message {
    fn from(val: CompletionMessage) -> Self {
        let mut content = vec![MessagePart::Text(TextPart {
            text: val.content.unwrap_or_default(),
        })];
        if let Some(tcs) = val.tool_calls {
            for tc in tcs {
                content.push(MessagePart::ToolCall(ToolCallPart {
                    id: tc.id,
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                }));
            }
        }
        Message {
            role: MessageRole::Assistant,
            content,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletion {
    index: usize,
    message: CompletionMessage,
    finish_reason: String,
}

/// Response body returned by the OpenAI chat completions endpoint.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIResponse {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub usage: OpenAIUsage,
    pub choices: Vec<ChatCompletion>,
}

impl From<OpenAIResponse> for LLMResponse {
    fn from(value: OpenAIResponse) -> Self {
        let message: Message = value
            .choices
            .first()
            .expect("LLM should have produced at least 1 message")
            .message
            .to_owned()
            .into();
        Self {
            message,
            usage: LLMUsage::from(value.usage),
            id: value.id,
            created_at: Some(value.created),
        }
    }
}

fn default_streaming_role() -> OpenAIMessageRole {
    OpenAIMessageRole::Assistant
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingFunction {
    #[serde(default)]
    pub name: Option<String>,
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingCompletionToolCall {
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default)]
    pub call_type: Option<String>,
    pub function: StreamingFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingCompletionDelta {
    #[serde(default = "default_streaming_role")]
    pub role: OpenAIMessageRole,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<StreamingCompletionToolCall>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingChatCompletion {
    pub index: u32,
    pub delta: StreamingCompletionDelta,
    pub finish_reason: Option<String>,
}

/// A single event emitted by the OpenAI streaming chat completions endpoint.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OpenAIStreamingMessage {
    pub id: String,
    pub created: u64,
    pub choices: Vec<StreamingChatCompletion>,
    pub usage: Option<OpenAISimplifiedUsage>,
}

impl From<OpenAIStreamingMessage> for LLMStreamingDelta {
    fn from(value: OpenAIStreamingMessage) -> Self {
        let (delta, is_stop) = value
            .choices
            .first()
            .map_or((Some(String::new()), true), |d| {
                (d.delta.content.clone(), d.finish_reason.is_some())
            });
        LLMStreamingDelta {
            response_id: value.id,
            created_at: Some(value.created),
            delta,
            stop: is_stop,
        }
    }
}

fn deltas_to_message(deltas: &[LLMStreamingDelta]) -> Message {
    let mut text = String::new();
    for delta in deltas {
        text += delta.delta.clone().unwrap_or_default().as_ref();
    }
    Message {
        role: MessageRole::Assistant,
        content: vec![MessagePart::Text(TextPart { text })],
    }
}

impl OpenAIClient {
    pub fn new(retry_policy: RetryPolicy) -> Self {
        Self { retry_policy }
    }

    pub async fn respond(
        &self,
        request: LLMRequest,
    ) -> Result<LLMResponse, Box<dyn std::error::Error>> {
        let base_url = request.base_url.clone().unwrap();
        let api_key = request.api_key.clone();
        if request.stream {
            return Err(StreamParamError {
                should_stream: false,
            }
            .into());
        }
        let req = OpenAIRequest::try_from(request)?;
        let retry = ExponentialBackoff::builder()
            .base(self.retry_policy.base)
            .jitter(self.retry_policy.jitter)
            .retry_bounds(
                self.retry_policy.min_retry_interval,
                self.retry_policy.max_retry_interval,
            )
            .build_with_max_retries(self.retry_policy.max_retries);
        let client = ClientBuilder::new(reqwest::Client::new())
            .with(RetryTransientMiddleware::new_with_policy(retry))
            .build();
        let body = serde_json::to_string(&req)?;
        let response = client
            .post(format!("{}{}", base_url, CHAT_COMPLETIONS_ENDPOINT))
            .bearer_auth(api_key)
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await?;

        let status = response.status();
        let text = response.text().await?;

        if !status.is_success() {
            // now you have the actual error body from the API
            return Err(format!("API error {}: {}", status, text).into());
        }

        let parsed: OpenAIResponse = serde_json::from_str(&text)?;

        Ok(LLMResponse::from(parsed))
    }

    pub async fn stream_response(
        &self,
        request: LLMRequest,
    ) -> Result<LLMStream, Box<dyn std::error::Error>> {
        let base_url = request.base_url.clone().unwrap();
        let api_key = request.api_key.clone();
        if !request.stream {
            return Err(StreamParamError {
                should_stream: true,
            }
            .into());
        }
        let req = OpenAIRequest::try_from(request)?;
        let retry = ExponentialBackoff::builder()
            .base(self.retry_policy.base)
            .jitter(self.retry_policy.jitter)
            .retry_bounds(
                self.retry_policy.min_retry_interval,
                self.retry_policy.max_retry_interval,
            )
            .build_with_max_retries(self.retry_policy.max_retries);
        let client = ClientBuilder::new(reqwest::Client::new())
            .with(RetryTransientMiddleware::new_with_policy(retry))
            .build();
        let body = serde_json::to_string(&req)?;
        let mut deltas: Vec<LLMStreamingDelta> = vec![];
        let mut response_id: Option<String> = None;
        let mut first_created: Option<u64> = None;
        let mut resp_usage: Option<LLMUsage> = None;
        let mut indexed_tool_calls: BTreeMap<u32, StreamingCompletionToolCall> = BTreeMap::new();

        let s: LLMStream = Box::pin(stream! {
            let first_response = client
                .post(format!("{}{}", base_url, CHAT_COMPLETIONS_ENDPOINT))
                .bearer_auth(api_key.clone())
                .header("Content-Type", "application/json")
                .body(body)
                .send()
                .await;
            let mut events;
            match first_response {
                Ok(r) => {
                    let status = r.status();
                    if !status.is_success() {
                        // only now consume the body, as text, for the error message
                        let text = r.text().await.unwrap_or("unknown error".to_string());
                        yield Err(format!("API error {}: {}", status, text).into());
                        return;
                    }
                    events = r.bytes_stream().eventsource();
                },
                Err(e) => {
                    yield Err(e.into());
                    return;
                }
            }
            while let Some(ev) = events.next().await {
                match ev {
                    Ok(event) => {
                        if event.data == "[DONE]" {
                            let message = deltas_to_message(&deltas);
                            let mut tool_calls = None;
                            if !indexed_tool_calls.is_empty() {
                                for (_, tc) in indexed_tool_calls.clone() {
                                    if !is_valid_json(&tc.function.arguments) {
                                        continue;
                                    }
                                    tool_calls.get_or_insert_with(Vec::new).push(ToolCallPart { id: tc.id.expect("The tool call should have been registered with an ID"), name: tc.function.name.expect("The tool call should have been registered with a function name"), arguments: tc.function.arguments });
                                }
                            }
                            yield Ok(LLMStreamingResponse::Complete(LLMStreamingComplete { id: response_id.clone().unwrap(), deltas: deltas.clone(), created_at: first_created, usage: resp_usage, message, tool_calls, thinking_deltas: None }))
                        } else {
                            let json_result: Result<OpenAIStreamingMessage, serde_json::Error> = serde_json::from_str(&event.data);
                            match json_result {
                                Ok(json) => {
                                    let streaming_delta = LLMStreamingDelta::from(json.clone());
                                    deltas.push(streaming_delta.clone());
                                    yield Ok(LLMStreamingResponse::Delta(streaming_delta));
                                    let tool_calls = json.choices.first().and_then(|c| c.delta.tool_calls.clone());
                                    if let Some(tcs) = tool_calls {
                                        for t in tcs {
                                            indexed_tool_calls.entry(t.index).and_modify(|v| v.function.arguments += &t.function.arguments).or_insert(t.clone());
                                            let indexd_tool_call = indexed_tool_calls.get(&t.index).unwrap();
                                            yield Ok(LLMStreamingResponse::ToolDelta(LLMToolDelta { tool_call_id: indexd_tool_call.id.clone().expect("Tool call should have been registered with an ID"), name: indexd_tool_call.function.name.clone().expect("Tool call should have been registered with a name"), partial_arguments: t.function.arguments }))
                                        }
                                    }
                                    if let Some(u) = json.usage {
                                        resp_usage = Some(LLMUsage::from(u));
                                    }
                                    if response_id.is_some() {

                                    } else {
                                        response_id = Some(json.id);
                                    }
                                    if first_created.is_some() {

                                    } else {
                                        first_created = Some(json.created);
                                    }
                                },
                                Err(err) => {
                                    yield Err(err.into());
                                    return;
                                }
                            }
                        }
                    },
                    Err(e) => {
                        yield Err(e.into());
                        return;
                    }
                }
            }
        });

        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ThinkingPart;

    fn text_message(role: MessageRole, text: &str) -> Message {
        Message {
            role,
            content: vec![MessagePart::Text(TextPart {
                text: text.to_string(),
            })],
        }
    }

    #[test]
    fn openai_message_user_role_maps_to_user() {
        let msg = text_message(MessageRole::User, "hello");
        let openai = OpenAIMessage::try_from(msg).unwrap();
        assert!(matches!(openai, OpenAIMessage::User(_)));
    }

    #[test]
    fn openai_message_system_role_maps_to_developer() {
        let msg = text_message(MessageRole::System, "sys");
        let openai = OpenAIMessage::try_from(msg).unwrap();
        assert!(matches!(openai, OpenAIMessage::Developer(_)));
    }

    #[test]
    fn openai_message_roundtrip_preserves_text() {
        let original = text_message(MessageRole::User, "roundtrip");
        let openai = OpenAIMessage::try_from(original.clone()).unwrap();
        let back: Message = openai.into();
        assert_eq!(back.role, MessageRole::User);
        assert_eq!(back.content.len(), 1);
        assert!(matches!(
            back.content.first(),
            Some(MessagePart::Text(TextPart { text })) if text == "roundtrip"
        ));
    }

    #[test]
    fn openai_message_rejects_thinking_part() {
        let msg = Message {
            role: MessageRole::Assistant,
            content: vec![MessagePart::Thinking(ThinkingPart::new("thinking"))],
        };
        let result = OpenAIMessage::try_from(msg);
        assert!(result.is_err());
    }

    #[test]
    fn openai_image_part_converts_to_data_url() {
        let part = ImagePart {
            data: "abc".to_string(),
            is_base64: true,
            mime_type: Some("image/png".to_string()),
        };
        let openai = OpenAIImagePart::from(part);
        assert_eq!(openai.part_type, "image_url");
        assert_eq!(openai.image_url.url, "data:image/png;base64,abc");
    }

    #[test]
    fn openai_image_part_converts_url_unchanged() {
        let part = ImagePart {
            data: "https://example.com/img.png".to_string(),
            is_base64: false,
            mime_type: None,
        };
        let openai = OpenAIImagePart::from(part);
        assert_eq!(openai.image_url.url, "https://example.com/img.png");
    }

    #[test]
    fn openai_audio_part_converts_mpeg_to_mp3() {
        let part = AudioPart {
            data: "d".to_string(),
            mime_type: "audio/mpeg".to_string(),
        };
        let openai = OpenAIAudioPart::from(part);
        assert_eq!(openai.part_type, "input_audio");
        assert_eq!(openai.input_audio.format, "mp3");
        assert_eq!(openai.input_audio.data, "d");
    }

    #[test]
    fn openai_audio_part_converts_wav_to_wav() {
        let part = AudioPart {
            data: "d".to_string(),
            mime_type: "audio/wav".to_string(),
        };
        let openai = OpenAIAudioPart::from(part);
        assert_eq!(openai.input_audio.format, "wav");
    }

    #[test]
    fn openai_message_into_message_preserves_image_data_url() {
        let openai = OpenAIMessage::User(OpenAISimpleMessage {
            role: OpenAIMessageRole::User,
            content: vec![OpenAIMessagePart::Image(OpenAIImagePart {
                part_type: "image_url".to_string(),
                image_url: OpenAIImageUrl {
                    url: "data:image/png;base64,abc".to_string(),
                },
            })],
        });
        let msg: Message = openai.into();
        assert!(matches!(
            msg.content.first(),
            Some(MessagePart::Image(ImagePart { data, is_base64: true, mime_type: Some(mt) }))
                if data == "abc" && mt == "image/png"
        ));
    }

    #[test]
    fn openai_message_into_message_preserves_image_url() {
        let openai = OpenAIMessage::User(OpenAISimpleMessage {
            role: OpenAIMessageRole::User,
            content: vec![OpenAIMessagePart::Image(OpenAIImagePart {
                part_type: "image_url".to_string(),
                image_url: OpenAIImageUrl {
                    url: "https://example.com/img.png".to_string(),
                },
            })],
        });
        let msg: Message = openai.into();
        assert!(matches!(
            msg.content.first(),
            Some(MessagePart::Image(ImagePart { data, is_base64: false, mime_type: None }))
                if data == "https://example.com/img.png"
        ));
    }

    #[test]
    fn openai_message_into_message_preserves_audio() {
        let openai = OpenAIMessage::User(OpenAISimpleMessage {
            role: OpenAIMessageRole::User,
            content: vec![OpenAIMessagePart::Audio(OpenAIAudioPart {
                part_type: "input_audio".to_string(),
                input_audio: OpenAIInputAudio {
                    data: "d".to_string(),
                    format: "mp3".to_string(),
                },
            })],
        });
        let msg: Message = openai.into();
        assert!(matches!(
            msg.content.first(),
            Some(MessagePart::Audio(AudioPart { data, mime_type }))
                if data == "d" && mime_type == "audio/mp3"
        ));
    }

    #[test]
    fn openai_message_into_message_assistant_preserves_tool_calls() {
        let openai = OpenAIMessage::Assistant(OpenAIAssistantMessage {
            role: OpenAIMessageRole::Assistant,
            content: vec![OpenAIMessagePart::Text(OpenAITextPart {
                part_type: "text".to_string(),
                text: "ok".to_string(),
            })],
            tool_calls: Some(vec![OpenAIToolCallPart {
                id: "call_1".to_string(),
                function: OpenAIFunctionCall {
                    name: "tool".to_string(),
                    arguments: "{}".to_string(),
                },
                part_type: "function".to_string(),
            }]),
        });
        let msg: Message = openai.into();
        assert_eq!(msg.content.len(), 2);
        assert!(matches!(
            msg.content.get(1),
            Some(MessagePart::ToolCall(ToolCallPart { id, name, arguments }))
                if id == "call_1" && name == "tool" && arguments == "{}"
        ));
    }

    #[test]
    fn openai_message_into_message_developer_keeps_system_role() {
        let openai = OpenAIMessage::Developer(OpenAISimpleMessage {
            role: OpenAIMessageRole::Developer,
            content: vec![OpenAIMessagePart::Text(OpenAITextPart {
                part_type: "text".to_string(),
                text: "sys".to_string(),
            })],
        });
        let msg: Message = openai.into();
        assert_eq!(msg.role, MessageRole::System);
    }

    #[test]
    fn openai_message_into_message_tool_maps_back() {
        let openai = OpenAIMessage::Tool(OpenAIToolResultMessage {
            role: OpenAIMessageRole::Tool,
            content: "result".to_string(),
            tool_call_id: "call_1".to_string(),
        });
        let msg: Message = openai.into();
        assert_eq!(msg.role, MessageRole::Tool);
        assert!(matches!(
            msg.content.first(),
            Some(MessagePart::ToolResult(ToolResultPart { tool_call_id, result }))
                if tool_call_id == "call_1" && result == "result"
        ));
    }

    #[test]
    fn openai_request_drops_empty_tools() {
        let request = LLMRequest {
            api_type: ApiType::OpenAI,
            base_url: Some("https://api.openai.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "gpt-4".to_string(),
            messages: vec![text_message(MessageRole::User, "hi")],
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            reasoning_effort: None,
            prompt_cache_ttl: None,
            stream: false,
            output_format: None,
            tools: Some(vec![]),
            tool_choice: None,
            parallel_tool_calls: false,
        };
        let openai_req = OpenAIRequest::try_from(request).unwrap();
        assert!(openai_req.tools.is_none());
        assert!(openai_req.parallel_tool_calls.is_none());
    }

    #[test]
    fn openai_request_enables_stream_options_when_streaming() {
        let mut request = LLMRequest {
            api_type: ApiType::OpenAI,
            base_url: Some("https://api.openai.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "gpt-4".to_string(),
            messages: vec![text_message(MessageRole::User, "hi")],
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            reasoning_effort: None,
            prompt_cache_ttl: None,
            stream: true,
            output_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
        };
        request.stream = true;
        let openai_req = OpenAIRequest::try_from(request).unwrap();
        assert!(openai_req.stream_options.is_some());
        assert!(openai_req.stream_options.unwrap().include_usage);
    }

    #[test]
    fn openai_usage_conversion_sums_other_tokens() {
        let usage = OpenAIUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            prompt_tokens_details: OpenAIPromptTokensDetails {
                cached_tokens: 2,
                audio_tokens: 1,
            },
            completion_tokens_details: OpenAICompletionTokensDetails {
                reasoning_tokens: 3,
                audio_tokens: 1,
                accepted_prediction_tokens: 0,
                rejected_prediction_tokens: 0,
            },
        };
        let llm_usage = LLMUsage::from(usage);
        assert_eq!(llm_usage.input_tokens, 10);
        assert_eq!(llm_usage.output_tokens, 5);
        assert_eq!(llm_usage.cache_read_tokens, Some(2));
        assert_eq!(llm_usage.other_tokens, Some(5));
    }

    #[test]
    fn openai_simplified_usage_conversion() {
        let usage = OpenAISimplifiedUsage {
            prompt_tokens: 7,
            completion_tokens: 3,
        };
        let llm_usage = LLMUsage::from(usage);
        assert_eq!(llm_usage.input_tokens, 7);
        assert_eq!(llm_usage.output_tokens, 3);
        assert!(llm_usage.cache_read_tokens.is_none());
        assert!(llm_usage.other_tokens.is_none());
    }

    #[test]
    fn openai_response_conversion_extracts_first_choice() {
        let response = OpenAIResponse {
            id: "resp_123".to_string(),
            created: 1712345678,
            model: "gpt-4".to_string(),
            usage: OpenAIUsage {
                prompt_tokens: 2,
                completion_tokens: 4,
                total_tokens: 6,
                prompt_tokens_details: OpenAIPromptTokensDetails {
                    cached_tokens: 0,
                    audio_tokens: 0,
                },
                completion_tokens_details: OpenAICompletionTokensDetails {
                    reasoning_tokens: 0,
                    audio_tokens: 0,
                    accepted_prediction_tokens: 0,
                    rejected_prediction_tokens: 0,
                },
            },
            choices: vec![ChatCompletion {
                index: 0,
                message: CompletionMessage {
                    role: OpenAIMessageRole::Assistant,
                    content: Some("hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: "stop".to_string(),
            }],
        };
        let llm_response = LLMResponse::from(response);
        assert_eq!(llm_response.id, "resp_123");
        assert_eq!(llm_response.created_at, Some(1712345678));
        assert_eq!(llm_response.usage.input_tokens, 2);
        assert_eq!(llm_response.usage.output_tokens, 4);
        assert!(matches!(
            llm_response.message.content.first(),
            Some(MessagePart::Text(TextPart { text })) if text == "hello"
        ));
    }

    #[test]
    fn openai_streaming_message_to_delta_stop_on_finish_reason() {
        let msg = OpenAIStreamingMessage {
            id: "stream_1".to_string(),
            created: 1712345678,
            choices: vec![StreamingChatCompletion {
                index: 0,
                delta: StreamingCompletionDelta {
                    role: OpenAIMessageRole::Assistant,
                    content: Some("ok".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        };
        let delta = LLMStreamingDelta::from(msg);
        assert_eq!(delta.response_id, "stream_1");
        assert_eq!(delta.delta, Some("ok".to_string()));
        assert!(delta.stop);
    }

    #[test]
    fn openai_streaming_message_to_delta_no_stop_without_finish_reason() {
        let msg = OpenAIStreamingMessage {
            id: "stream_2".to_string(),
            created: 1712345678,
            choices: vec![StreamingChatCompletion {
                index: 0,
                delta: StreamingCompletionDelta {
                    role: OpenAIMessageRole::Assistant,
                    content: Some("more".to_string()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let delta = LLMStreamingDelta::from(msg);
        assert_eq!(delta.delta, Some("more".to_string()));
        assert!(!delta.stop);
    }

    #[test]
    fn deltas_to_message_joins_text() {
        let deltas = vec![
            LLMStreamingDelta {
                response_id: "a".to_string(),
                created_at: None,
                delta: Some("hello ".to_string()),
                stop: false,
            },
            LLMStreamingDelta {
                response_id: "a".to_string(),
                created_at: None,
                delta: Some("world".to_string()),
                stop: true,
            },
        ];
        let message = deltas_to_message(&deltas);
        assert_eq!(message.role, MessageRole::Assistant);
        assert_eq!(message.content.len(), 1);
        let joined: String = message
            .content
            .into_iter()
            .filter_map(|p| match p {
                MessagePart::Text(t) => Some(t.text),
                _ => None,
            })
            .collect();
        assert_eq!(joined, "hello world");
    }
}
