use futures_util::StreamExt;
use std::{collections::BTreeMap, fmt::Display, str::FromStr};

use async_stream::stream;
use eventsource_stream::Eventsource;
use reqwest_middleware::ClientBuilder;
use reqwest_retry::{RetryTransientMiddleware, policies::ExponentialBackoff};
use schemars::Schema;
use serde::{Deserialize, Serialize};

use crate::{
    ApiType, DocumentPart, ImagePart, LLMRequest, LLMResponse, LLMStream, LLMStreamingComplete,
    LLMStreamingDelta, LLMStreamingResponse, LLMThinkingDelta, LLMToolDelta, LLMUsage,
    MESSAGES_ENDPOINT, Message, MessagePart, MessageRole, ReasoningEffort, RetryPolicy, TextPart,
    ThinkingPart, Tool, ToolCallPart, ToolChoice, ToolResultPart,
    errors::{
        InvalidAntRequestConversion, InvalidTtl, StreamParamError, UnsupportedPartType,
        UnsupportedType,
    },
    is_valid_json,
};

pub const DEFAULT_MAX_TOKENS: u32 = 128_000;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum AntMessageRole {
    System,
    User,
    Assistant,
}

impl From<MessageRole> for AntMessageRole {
    fn from(value: MessageRole) -> Self {
        match value {
            MessageRole::Assistant => Self::Assistant,
            MessageRole::Tool => Self::User,
            MessageRole::System => Self::System,
            MessageRole::User => Self::User,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub cache_type: String,
    pub ttl: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntTextPart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub text: String,
}

impl From<TextPart> for AntTextPart {
    fn from(value: TextPart) -> Self {
        Self {
            part_type: "text".to_string(),
            text: value.text,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Base64ImageSource {
    pub data: String,
    pub media_type: String,
    #[serde(rename = "type")]
    pub source_type: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct URLImageSource {
    pub url: String,
    #[serde(rename = "type")]
    pub source_type: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum ImageSource {
    Base64(Base64ImageSource),
    URL(URLImageSource),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntImagePart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub source: ImageSource,
}

impl From<ImagePart> for AntImagePart {
    fn from(value: ImagePart) -> Self {
        let source = if value.is_base64 {
            ImageSource::Base64(Base64ImageSource {
                data: value.data,
                media_type: value
                    .mime_type
                    .expect("base64 data are expected to have a mime_type"),
                source_type: "base64".to_string(),
            })
        } else {
            ImageSource::URL(URLImageSource {
                url: value.data,
                source_type: "url".to_string(),
            })
        };
        Self {
            part_type: "image".to_string(),
            source,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntToolCallPart {
    id: String,
    name: String,
    input: serde_json::Map<String, serde_json::Value>,
    #[serde(rename = "type")]
    part_type: String,
}

impl From<ToolCallPart> for AntToolCallPart {
    fn from(value: ToolCallPart) -> Self {
        Self {
            id: value.id,
            name: value.name,
            input: serde_json::from_str(&value.arguments)
                .expect("Arguments should be JSON serializable"),
            part_type: "tool_use".to_string(),
        }
    }
}

impl From<AntToolCallPart> for ToolCallPart {
    fn from(val: AntToolCallPart) -> Self {
        ToolCallPart {
            id: val.id,
            name: val.name,
            arguments: serde_json::to_string(&val.input).expect("Should be serializable"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntToolResultPart {
    pub tool_use_id: String,
    pub content: String,
    #[serde(rename = "type")]
    pub part_type: String,
}

impl From<ToolResultPart> for AntToolResultPart {
    fn from(value: ToolResultPart) -> Self {
        Self {
            tool_use_id: value.tool_call_id,
            content: value.result,
            part_type: "tool_result".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntThinkingPart {
    pub thinking: String,
    #[serde(default)]
    pub signature: Option<String>,
    #[serde(rename = "type")]
    pub part_type: String,
}

impl From<ThinkingPart> for AntThinkingPart {
    fn from(value: ThinkingPart) -> Self {
        Self {
            thinking: value.thinking,
            signature: value.signature,
            part_type: "thinking".into(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Base64PDFDocumentSource {
    pub data: String,
    pub media_type: String,
    #[serde(rename = "type")]
    pub source_type: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PlainTextDocumentSource {
    pub data: String,
    pub media_type: String,
    #[serde(rename = "type")]
    pub source_type: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UrlPdfDocumentSource {
    pub url: String,
    #[serde(rename = "type")]
    pub source_type: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum DocumentSource {
    Base64PDF(Base64PDFDocumentSource),
    PlainText(PlainTextDocumentSource),
    UrlPdf(UrlPdfDocumentSource),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntDocumentPart {
    pub source: DocumentSource,
    #[serde(rename = "type")]
    pub part_type: String,
}

impl From<DocumentPart> for AntDocumentPart {
    fn from(value: DocumentPart) -> Self {
        let (data, is_text, is_url) = match value.mime_type {
            Some(m) => match m.as_str() {
                "application/pdf" => (value.data, false, false),
                "text/plain" => (value.data, true, false),
                _ => unreachable!("This branch should not be reached"),
            },
            None => (value.data, false, true),
        };
        if is_url {
            Self {
                source: DocumentSource::UrlPdf(UrlPdfDocumentSource {
                    url: data,
                    source_type: "url".to_string(),
                }),
                part_type: "document".to_string(),
            }
        } else if is_text {
            Self {
                source: DocumentSource::PlainText(PlainTextDocumentSource {
                    data,
                    media_type: "text/plain".to_string(),
                    source_type: "text".to_string(),
                }),
                part_type: "document".to_string(),
            }
        } else {
            Self {
                source: DocumentSource::Base64PDF(Base64PDFDocumentSource {
                    data,
                    media_type: "application/pdf".to_string(),
                    source_type: "base64".to_string(),
                }),
                part_type: "document".to_string(),
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum AntMessagePart {
    Text(AntTextPart),
    Thinking(AntThinkingPart),
    Image(AntImagePart),
    Document(AntDocumentPart),
    ToolCall(AntToolCallPart),
    ToolResult(AntToolResultPart),
}

impl From<AntMessagePart> for MessagePart {
    fn from(val: AntMessagePart) -> Self {
        match val {
            AntMessagePart::Text(t) => MessagePart::Text(TextPart { text: t.text }),
            AntMessagePart::Document(d) => {
                let (data, is_base64, mime_type) = match d.source {
                    DocumentSource::Base64PDF(b) => (b.data, true, Some(b.media_type)),
                    DocumentSource::UrlPdf(u) => (u.url, false, None),
                    DocumentSource::PlainText(t) => (t.data, false, Some(t.media_type)),
                };
                MessagePart::Document(DocumentPart {
                    data,
                    mime_type,
                    is_base64,
                })
            }
            AntMessagePart::Image(i) => {
                let (data, is_base64, mime_type) = match i.source {
                    ImageSource::Base64(b) => (b.data, true, Some(b.media_type)),
                    ImageSource::URL(u) => (u.url, false, None),
                };
                MessagePart::Image(ImagePart {
                    data,
                    is_base64,
                    mime_type,
                })
            }
            AntMessagePart::Thinking(t) => MessagePart::Thinking(ThinkingPart {
                thinking: t.thinking,
                signature: t.signature,
            }),
            AntMessagePart::ToolCall(tc) => MessagePart::ToolCall(ToolCallPart {
                id: tc.id,
                name: tc.name,
                arguments: serde_json::to_string(&tc.input)
                    .expect("Tool input should be serializable"),
            }),
            AntMessagePart::ToolResult(tr) => MessagePart::ToolResult(ToolResultPart {
                tool_call_id: tr.tool_use_id,
                result: tr.content,
            }),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntMessage {
    role: AntMessageRole,
    content: Vec<AntMessagePart>,
}

impl TryFrom<Message> for AntMessage {
    type Error = UnsupportedPartType;

    fn try_from(value: Message) -> Result<Self, Self::Error> {
        let role = AntMessageRole::from(value.role);
        let mut content = vec![];
        for p in value.content {
            match p {
                MessagePart::Text(t) => {
                    content.push(AntMessagePart::Text(AntTextPart::from(t)));
                }
                MessagePart::Document(d) => {
                    content.push(AntMessagePart::Document(AntDocumentPart::from(d)));
                }
                MessagePart::Thinking(t) => {
                    content.push(AntMessagePart::Thinking(AntThinkingPart::from(t)));
                }
                MessagePart::Image(i) => {
                    content.push(AntMessagePart::Image(AntImagePart::from(i)));
                }
                MessagePart::ToolCall(tc) => {
                    content.push(AntMessagePart::ToolCall(AntToolCallPart::from(tc)));
                }
                MessagePart::ToolResult(tr) => {
                    content.push(AntMessagePart::ToolResult(AntToolResultPart::from(tr)));
                }
                MessagePart::Audio(_) => {
                    return Err(UnsupportedPartType {
                        part_type: "audio".to_string(),
                        api_type: ApiType::Anthropic.to_string(),
                    });
                }
            }
        }
        Ok(Self { role, content })
    }
}

pub enum AllowedCacheTtl {
    FiveMinutes,
    OneHour,
}

impl FromStr for AllowedCacheTtl {
    type Err = InvalidTtl;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "5m" => Ok(Self::FiveMinutes),
            "1h" => Ok(Self::OneHour),
            _ => Err(InvalidTtl { ttl: s.to_string() }),
        }
    }
}

impl Display for AllowedCacheTtl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FiveMinutes => write!(f, "5m"),
            Self::OneHour => write!(f, "1h"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntAutoToolChoice {
    #[serde(rename = "type")]
    pub tool_choice_type: String,
    pub disable_parallel_tool_use: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntAnyToolChoice {
    #[serde(rename = "type")]
    pub tool_choice_type: String,
    pub disable_parallel_tool_use: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntNoneToolChoice {
    #[serde(rename = "type")]
    pub tool_choice_type: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum AntToolChoice {
    Auto(AntAutoToolChoice),
    Any(AntAnyToolChoice),
    None(AntNoneToolChoice),
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum AntEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl TryFrom<ReasoningEffort> for AntEffort {
    type Error = UnsupportedType;
    fn try_from(value: ReasoningEffort) -> Result<Self, Self::Error> {
        match value {
            ReasoningEffort::Low => Ok(Self::Low),
            ReasoningEffort::Medium => Ok(Self::Medium),
            ReasoningEffort::High => Ok(Self::High),
            ReasoningEffort::Minimal => Ok(Self::Low),
            ReasoningEffort::Xhigh => Ok(Self::Xhigh),
            ReasoningEffort::Maximum => Ok(Self::Max),
            _ => Err(UnsupportedType {
                unsupported_type: "none".to_string(),
            }),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntOutputFormat {
    #[serde(rename = "type")]
    format_type: String,
    schema: Schema,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntOutputConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<AntEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<AntOutputFormat>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntTool {
    pub name: String,
    pub description: String,
    pub input_schema: Schema,
}

impl From<Tool> for AntTool {
    fn from(value: Tool) -> Self {
        Self {
            name: value.name,
            description: value.description,
            input_schema: value.parameters,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntThinkingConfigDisabled {
    #[serde(rename = "type")]
    config_type: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntThinkingConfigAdaptive {
    #[serde(rename = "type")]
    config_type: String,
    display: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum AntThinkingConfig {
    Disabled(AntThinkingConfigDisabled),
    Adaptive(AntThinkingConfigAdaptive),
}

/// Serialized request body sent to the Anthropic Messages API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntRequest {
    pub model: String,
    pub messages: Vec<AntMessage>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<AntOutputConfig>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AntToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AntTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AntThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
}

impl TryFrom<LLMRequest> for AntRequest {
    type Error = InvalidAntRequestConversion;

    fn try_from(value: LLMRequest) -> Result<Self, Self::Error> {
        let mut messages = vec![];
        let mut system = None;
        for m in value.messages {
            let message = AntMessage::try_from(m)?;
            match message.role {
                AntMessageRole::System => {
                    let mut text = String::new();
                    for p in message.content {
                        match p {
                            AntMessagePart::Text(t) => text += &t.text,
                            _ => continue,
                        }
                    }
                    system = Some(text)
                }
                _ => messages.push(message),
            }
        }
        let cache_control = match value.prompt_cache_ttl {
            Some(p) => Some(CacheControl {
                ttl: AllowedCacheTtl::from_str(&p)?.to_string(),
                cache_type: "ephemeral".to_string(),
            }),
            None => None,
        };
        let mut tool_choice = None;
        let tools: Option<Vec<AntTool>> = if let Some(ts) = value.tools
            && !ts.is_empty()
        {
            tool_choice = value.tool_choice.map(|tc| match tc {
                ToolChoice::Auto => AntToolChoice::Auto(AntAutoToolChoice {
                    tool_choice_type: "auto".to_string(),
                    disable_parallel_tool_use: !value.parallel_tool_calls,
                }),
                ToolChoice::Required => AntToolChoice::Any(AntAnyToolChoice {
                    tool_choice_type: "any".to_string(),
                    disable_parallel_tool_use: !value.parallel_tool_calls,
                }),
                ToolChoice::None => AntToolChoice::None(AntNoneToolChoice {
                    tool_choice_type: "none".to_string(),
                }),
            });
            Some(ts.iter().cloned().map(AntTool::from).collect())
        } else {
            None
        };
        let effort = if let Some(re) = value.reasoning_effort {
            AntEffort::try_from(re).ok()
        } else {
            None
        };
        let output_format = value.output_format.map(|f| AntOutputFormat {
            format_type: "json_schema".to_string(),
            schema: {
                if f.schema
                    .get("additionalProperties")
                    .is_none_or(|f| f.as_bool().is_none() || f.as_bool().is_some_and(|o| o))
                {
                    let mut schema = f.schema;
                    schema.insert(
                        "additionalProperties".to_string(),
                        serde_json::Value::from(false),
                    );
                    schema
                } else {
                    f.schema
                }
            },
        });
        let output_config = if output_format.is_some() || effort.is_some() {
            Some(AntOutputConfig {
                effort,
                format: output_format,
            })
        } else {
            None
        };
        let thinking = effort.map_or(
            AntThinkingConfig::Disabled(AntThinkingConfigDisabled {
                config_type: "disabled".to_string(),
            }),
            |_| {
                AntThinkingConfig::Adaptive(AntThinkingConfigAdaptive {
                    config_type: "adaptive".to_string(),
                    display: "summarized".to_string(),
                })
            },
        );
        let top_p = if let Some(tp) = value.top_p {
            if value.temperature.is_some() {
                None
            } else if matches!(thinking, AntThinkingConfig::Adaptive(_)) {
                Some(0.95)
            } else {
                Some(tp)
            }
        } else {
            None
        };
        let temperature = value.temperature.map(|p| {
            if matches!(thinking, AntThinkingConfig::Adaptive(_)) {
                1_f32
            } else {
                p
            }
        });
        Ok(Self {
            model: value.model,
            max_tokens: value.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            messages,
            stream: value.stream,
            cache_control,
            tool_choice,
            tools,
            output_config,
            thinking: Some(thinking),
            temperature,
            top_p,
            system,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct AntUsage {
    pub input_tokens: u32,
    pub cache_creation_input_tokens: u32,
    pub cache_read_input_tokens: u32,
    pub output_tokens: u32,
}

impl From<AntUsage> for LLMUsage {
    fn from(value: AntUsage) -> Self {
        Self {
            input_tokens: value.input_tokens,
            cache_read_tokens: Some(value.cache_read_input_tokens),
            cache_write_tokens: Some(value.cache_creation_input_tokens),
            output_tokens: value.output_tokens,
            other_tokens: None,
        }
    }
}

/// Response body returned by the Anthropic Messages API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntResponse {
    pub id: String,
    pub content: Vec<AntMessagePart>,
    pub usage: AntUsage,
}

impl From<AntResponse> for LLMResponse {
    fn from(value: AntResponse) -> Self {
        let mut content: Vec<MessagePart> = vec![];
        for c in value.content {
            content.push(c.into());
        }
        Self {
            id: value.id,
            message: Message {
                role: MessageRole::Assistant,
                content,
            },
            created_at: None,
            usage: LLMUsage::from(value.usage),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamEventType {
    MessageStart,
    ContentBlockStart,
    ContentBlockDelta,
    ContentBlockStop,
    MessageDelta,
    MessageStop,
    Ping,
    Error,
}

impl FromStr for StreamEventType {
    type Err = UnsupportedType;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "message_start" => Ok(Self::MessageStart),
            "message_delta" => Ok(Self::MessageDelta),
            "message_stop" => Ok(Self::MessageStop),
            "content_block_start" => Ok(Self::ContentBlockStart),
            "content_block_delta" => Ok(Self::ContentBlockDelta),
            "content_block_stop" => Ok(Self::ContentBlockStop),
            "error" => Ok(Self::Error),
            "ping" => Ok(Self::Ping),
            _ => Err(UnsupportedType {
                unsupported_type: s.to_string(),
            }),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamTextDelta {
    #[serde(rename = "type")]
    pub delta_type: String,
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamToolDelta {
    #[serde(rename = "type")]
    pub delta_type: String,
    pub partial_json: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamThinkingDelta {
    #[serde(rename = "type")]
    pub delta_type: String,
    pub thinking: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamThinkingSignatureDelta {
    #[serde(rename = "type")]
    pub delta_type: String,
    pub signature: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamTextBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub index: u32,
    pub delta: StreamTextDelta,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamToolBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub index: u32,
    pub delta: StreamToolDelta,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamThinkingBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub index: u32,
    pub delta: StreamThinkingDelta,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamThinkingSignatureBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub index: u32,
    pub delta: StreamThinkingSignatureDelta,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum StreamContentBlock {
    Text(StreamTextBlock),
    Tool(StreamToolBlock),
    Thinking(StreamThinkingBlock),
    ThinkingSignature(StreamThinkingSignatureBlock),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamContentStart {
    #[serde(rename = "type")]
    pub content_type: String,
    pub content_block: AntMessagePart,
    pub index: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamContentStop {
    #[serde(rename = "type")]
    pub content_type: String,
    pub index: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamMessageStart {
    #[serde(rename = "type")]
    message_type: String,
    message: AntResponse,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamMessageDelta {
    #[serde(rename = "type")]
    message_type: String,
    usage: AntUsage,
}

/// Client for sending requests to the Anthropic Messages API.
#[derive(Debug, Clone, Copy, Default)]
pub struct AntClient {
    /// Retry policy applied to API requests made by this client.
    pub retry_policy: RetryPolicy,
}

impl AntClient {
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
        let req = AntRequest::try_from(request)?;
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
            .post(format!("{}{}", base_url, MESSAGES_ENDPOINT))
            .header("X-Api-Key", api_key)
            .header("anthropic-version", "2023-06-01")
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

        let parsed: AntResponse = serde_json::from_str(&text)?;

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
        let req = AntRequest::try_from(request)?;
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
        let mut thinking_deltas: Vec<LLMThinkingDelta> = vec![];
        let mut response_id: Option<String> = None;
        let mut resp_usage: Option<LLMUsage> = None;
        let mut indexed_tool_calls: BTreeMap<u32, ToolCallPart> = BTreeMap::new();
        let mut parts: BTreeMap<u32, MessagePart> = BTreeMap::new();

        let s: LLMStream = Box::pin(stream! {
            let response_result = client
                .post(format!("{}{}", base_url, MESSAGES_ENDPOINT))
                .header("X-Api-Key", api_key)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .body(body)
                .send()
                .await;

            let mut events;

            match response_result {
                Ok(response) => {
                    let status = response.status();
                    if !status.is_success() {
                        // only now consume the body, as text, for the error message
                        let text = response.text().await.unwrap_or_default();
                        yield Err(format!("API error {}: {}", status, text).into());
                        return;
                    }
                    events = response.bytes_stream().eventsource();
                },
                Err(e) => {
                    yield Err(e.into());
                    return;
                }
            }

            while let Some(ev) = events.next().await {
                match ev {
                    Err(e) => {
                        yield Err(e.into());
                        return;
                    }
                    Ok(event) => {
                        let event_result = StreamEventType::from_str(&event.event);
                        let event_type = match event_result {
                            Err(e) => {
                                yield Err(e.into());
                                return;
                            },
                            Ok(t) => {
                                t
                            }
                        };
                        match event_type {
                            StreamEventType::MessageStart => {
                                let res: Result<StreamMessageStart, serde_json::Error> = serde_json::from_str(&event.data);
                                match res {
                                    Err(e) => {
                                        yield Err(e.into());
                                        return;
                                    }
                                    Ok(m) => {
                                        response_id = Some(m.message.id)
                                    }
                                }
                            },
                            StreamEventType::MessageDelta => {
                                let res: Result<StreamMessageDelta, serde_json::Error> = serde_json::from_str(&event.data);
                                match res {
                                    Err(e) => {
                                        yield Err(e.into());
                                        return;
                                    }
                                    Ok(m) => {
                                        resp_usage = Some(LLMUsage::from(m.usage))
                                    }
                                }
                            },
                            StreamEventType::MessageStop => {
                                let mut tool_calls = None;
                                if !indexed_tool_calls.is_empty() {
                                    for (idx, tc) in indexed_tool_calls.clone() {
                                        if !is_valid_json(&tc.arguments) {
                                            continue;
                                        }
                                        parts.insert(idx, MessagePart::ToolCall(tc.to_owned()));
                                        tool_calls.get_or_insert_with(Vec::new).push(tc);
                                    }
                                }
                                yield Ok(LLMStreamingResponse::Complete(LLMStreamingComplete {
                                    id: response_id.clone().expect("response ID should be known by now"),
                                    created_at: None,
                                    thinking_deltas: Some(thinking_deltas.clone()),
                                    deltas: deltas.clone(),
                                    tool_calls,
                                    usage: resp_usage,
                                    message: Message { role: MessageRole::Assistant, content: parts.values().map(|p| p.to_owned()).collect() }
                                }));
                                break;
                            },
                            StreamEventType::Error => {
                                let res: Result<StreamError, serde_json::Error> = serde_json::from_str(&event.data);
                                match res {
                                    Err(e) => {
                                        yield Err(e.into());
                                        return;
                                    }
                                    Ok(m) => {
                                        yield Err(m.error.into());
                                        return;
                                    }
                                }
                            },
                            StreamEventType::Ping => {
                                continue;
                            },
                            StreamEventType::ContentBlockStart => {
                                let res: Result<StreamContentStart, serde_json::Error> = serde_json::from_str(&event.data);
                                match res {
                                    Err(e) => {
                                        yield Err(e.into());
                                        return;
                                    }
                                    Ok(st) => {
                                        match st.content_block {
                                            AntMessagePart::ToolCall(tc) => {
                                                indexed_tool_calls.insert(st.index, ToolCallPart { id: tc.id, name: tc.name, arguments: String::new() });
                                            },
                                            AntMessagePart::Thinking(t) => {
                                                parts.insert(st.index, MessagePart::Thinking(ThinkingPart { thinking: t.thinking, signature: t.signature }));
                                            },
                                            AntMessagePart::Text(t) => {
                                                parts.insert(st.index, MessagePart::Text(TextPart { text: t.text }));
                                            },
                                            // Assistant shouldn't produce any other part, safe to ignore
                                            _ => continue,
                                        }
                                    }
                                }
                            },
                            StreamEventType::ContentBlockDelta => {
                                let res: Result<StreamContentBlock, serde_json::Error> = serde_json::from_str(&event.data);
                                match res {
                                    Err(e) => {
                                        yield Err(e.into());
                                        return;
                                    }
                                    Ok(c) => {
                                        match c {
                                            StreamContentBlock::Text(t) => {
                                                parts.entry(t.index).and_modify(|v| {
                                                    if let MessagePart::Text(text) = v {
                                                        text.text += &t.delta.text;
                                                    }
                                                });
                                                let stream_delta = LLMStreamingDelta {
                                                    response_id: response_id.clone().expect("Response ID should be set by now"),
                                                    created_at: None,
                                                    delta: Some(t.delta.text),
                                                    stop: false,
                                                };
                                                deltas.push(stream_delta.clone());
                                                yield Ok(LLMStreamingResponse::Delta(stream_delta));
                                            }
                                            StreamContentBlock::Thinking(t) => {
                                                parts.entry(t.index).and_modify(|v| {
                                                    if let MessagePart::Thinking(think) = v {
                                                        think.thinking += &t.delta.thinking;
                                                    }
                                                });
                                                let thinking_delta = LLMThinkingDelta {
                                                    response_id: response_id.clone().expect("Response ID should be set by now"),
                                                    created_at: None,
                                                    delta: Some(t.delta.thinking),
                                                };
                                                thinking_deltas.push(thinking_delta.clone());
                                                yield Ok(LLMStreamingResponse::ThinkingDelta(thinking_delta));
                                            },
                                            StreamContentBlock::Tool(t) => {
                                                indexed_tool_calls.entry(t.index).and_modify(|e| e.arguments += &t.delta.partial_json);
                                                let tool_call = indexed_tool_calls.get(&t.index).expect("Tool call should have been registered by now");
                                                let tool_delta = LLMToolDelta {
                                                    response_id: response_id.clone().expect("Response ID should be set by now"),
                                                    tool_call_id: tool_call.id.clone(),
                                                    partial_arguments: t.delta.partial_json,
                                                    name: tool_call.name.clone(),
                                                };
                                                yield Ok(LLMStreamingResponse::ToolDelta(tool_delta))
                                            },
                                            StreamContentBlock::ThinkingSignature(ts) => {
                                                parts.entry(ts.index).and_modify(|v| {
                                                    if let MessagePart::Thinking(think) = v {
                                                        think.signature.get_or_insert_with(String::new).push_str(&ts.delta.signature);
                                                    }
                                                });
                                            }
                                        }
                                    }
                                }

                            },
                            StreamEventType::ContentBlockStop => continue
                        }
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
    use crate::{AudioPart, OutputFormat};

    fn text_message(role: MessageRole, text: &str) -> Message {
        Message {
            role,
            content: vec![MessagePart::Text(TextPart {
                text: text.to_string(),
            })],
        }
    }

    #[test]
    fn ant_message_user_role_maps_to_user() {
        let msg = text_message(MessageRole::User, "hello");
        let ant = AntMessage::try_from(msg).unwrap();
        assert!(matches!(ant.role, AntMessageRole::User));
        assert_eq!(ant.content.len(), 1);
    }

    #[test]
    fn ant_message_system_role_maps_to_system() {
        let msg = text_message(MessageRole::System, "sys");
        let ant = AntMessage::try_from(msg).unwrap();
        assert!(matches!(ant.role, AntMessageRole::System));
    }

    #[test]
    fn ant_message_rejects_audio_part() {
        let msg = Message {
            role: MessageRole::User,
            content: vec![MessagePart::Audio(AudioPart {
                data: "data".to_string(),
                mime_type: "audio/wav".to_string(),
            })],
        };
        let result = AntMessage::try_from(msg);
        assert!(result.is_err());
    }

    #[test]
    fn ant_message_part_roundtrip_text() {
        let _original = MessagePart::Text(TextPart {
            text: "roundtrip".to_string(),
        });
        let ant_part = AntMessagePart::Text(AntTextPart {
            part_type: "text".to_string(),
            text: "roundtrip".to_string(),
        });
        let back: MessagePart = ant_part.into();
        assert!(matches!(
            back,
            MessagePart::Text(TextPart { text }) if text == "roundtrip"
        ));
    }

    #[test]
    fn ant_message_part_roundtrip_image_base64() {
        let ant_part = AntMessagePart::Image(AntImagePart {
            part_type: "image".to_string(),
            source: ImageSource::Base64(Base64ImageSource {
                data: "abc".to_string(),
                media_type: "image/png".to_string(),
                source_type: "base64".to_string(),
            }),
        });
        let back: MessagePart = ant_part.into();
        assert!(matches!(
            back,
            MessagePart::Image(ImagePart { data, is_base64: true, mime_type: Some(mt) })
                if data == "abc" && mt == "image/png"
        ));
    }

    #[test]
    fn ant_message_part_roundtrip_image_url() {
        let ant_part = AntMessagePart::Image(AntImagePart {
            part_type: "image".to_string(),
            source: ImageSource::URL(URLImageSource {
                url: "https://example.com/img.png".to_string(),
                source_type: "url".to_string(),
            }),
        });
        let back: MessagePart = ant_part.into();
        assert!(matches!(
            back,
            MessagePart::Image(ImagePart { data, is_base64: false, mime_type: None })
                if data == "https://example.com/img.png"
        ));
    }

    #[test]
    fn ant_request_extracts_system_message() {
        let request = LLMRequest {
            api_type: ApiType::Anthropic,
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "claude".to_string(),
            messages: vec![
                text_message(MessageRole::System, "system prompt"),
                text_message(MessageRole::User, "hi"),
            ],
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            reasoning_effort: None,
            prompt_cache_ttl: None,
            stream: false,
            output_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
        };
        let ant_req = AntRequest::try_from(request).unwrap();
        assert_eq!(ant_req.system, Some("system prompt".to_string()));
        assert_eq!(ant_req.messages.len(), 1);
    }

    #[test]
    fn ant_request_default_max_tokens_when_unset() {
        let request = LLMRequest {
            api_type: ApiType::Anthropic,
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "claude".to_string(),
            messages: vec![text_message(MessageRole::User, "hi")],
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            reasoning_effort: None,
            prompt_cache_ttl: None,
            stream: false,
            output_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
        };
        let ant_req = AntRequest::try_from(request).unwrap();
        assert_eq!(ant_req.max_tokens, DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn ant_request_disables_top_p_when_temperature_present() {
        let request = LLMRequest {
            api_type: ApiType::Anthropic,
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "claude".to_string(),
            messages: vec![text_message(MessageRole::User, "hi")],
            max_output_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            reasoning_effort: None,
            prompt_cache_ttl: None,
            stream: false,
            output_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
        };
        let ant_req = AntRequest::try_from(request).unwrap();
        assert!(ant_req.top_p.is_none());
        assert_eq!(ant_req.temperature, Some(0.7));
    }

    #[test]
    fn ant_request_defaults_top_p_to_0_95() {
        let request = LLMRequest {
            api_type: ApiType::Anthropic,
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "claude".to_string(),
            messages: vec![text_message(MessageRole::User, "hi")],
            max_output_tokens: Some(100),
            temperature: None,
            top_p: Some(0.9),
            reasoning_effort: Some(ReasoningEffort::Medium),
            prompt_cache_ttl: None,
            stream: false,
            output_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
        };
        let ant_req = AntRequest::try_from(request).unwrap();
        assert_eq!(ant_req.top_p, Some(0.95));
    }

    #[test]
    fn ant_request_defaults_temperature_to_1() {
        let request = LLMRequest {
            api_type: ApiType::Anthropic,
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "claude".to_string(),
            messages: vec![text_message(MessageRole::User, "hi")],
            max_output_tokens: Some(100),
            temperature: Some(0.5),
            top_p: None,
            reasoning_effort: Some(ReasoningEffort::Medium),
            prompt_cache_ttl: None,
            stream: false,
            output_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
        };
        let ant_req = AntRequest::try_from(request).unwrap();
        assert_eq!(ant_req.temperature, Some(1_f32));
    }

    #[test]
    fn ant_usage_conversion_maps_cache_tokens() {
        let usage = AntUsage {
            input_tokens: 10,
            cache_creation_input_tokens: 3,
            cache_read_input_tokens: 2,
            output_tokens: 5,
        };
        let llm_usage = LLMUsage::from(usage);
        assert_eq!(llm_usage.input_tokens, 10);
        assert_eq!(llm_usage.output_tokens, 5);
        assert_eq!(llm_usage.cache_read_tokens, Some(2));
        assert_eq!(llm_usage.cache_write_tokens, Some(3));
    }

    #[test]
    fn ant_request_sets_cache_control_for_valid_ttl() {
        let request = LLMRequest {
            api_type: ApiType::Anthropic,
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "claude".to_string(),
            messages: vec![text_message(MessageRole::User, "hi")],
            max_output_tokens: Some(100),
            temperature: None,
            top_p: None,
            reasoning_effort: None,
            prompt_cache_ttl: Some("5m".to_string()),
            stream: false,
            output_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
        };
        let ant_req = AntRequest::try_from(request).unwrap();
        let cache = ant_req.cache_control.unwrap();
        assert_eq!(cache.cache_type, "ephemeral");
        assert_eq!(cache.ttl, "5m");
    }

    #[test]
    fn ant_request_rejects_invalid_cache_ttl() {
        let request = LLMRequest {
            api_type: ApiType::Anthropic,
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "claude".to_string(),
            messages: vec![text_message(MessageRole::User, "hi")],
            max_output_tokens: Some(100),
            temperature: None,
            top_p: None,
            reasoning_effort: None,
            prompt_cache_ttl: Some("10s".to_string()),
            stream: false,
            output_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
        };
        assert!(AntRequest::try_from(request).is_err());
    }

    #[test]
    fn ant_request_sets_effort_in_output_config() {
        let request = LLMRequest {
            api_type: ApiType::Anthropic,
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "claude".to_string(),
            messages: vec![text_message(MessageRole::User, "hi")],
            max_output_tokens: Some(100),
            temperature: None,
            top_p: None,
            reasoning_effort: Some(ReasoningEffort::High),
            prompt_cache_ttl: None,
            stream: false,
            output_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
        };
        let ant_req = AntRequest::try_from(request).unwrap();
        let output_config = ant_req.output_config.unwrap();
        assert!(matches!(output_config.effort, Some(AntEffort::High)));
    }

    #[test]
    fn ant_request_output_format_sets_additional_properties_false() {
        let schema: schemars::Schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        })
        .try_into()
        .unwrap();
        let request = LLMRequest {
            api_type: ApiType::Anthropic,
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "claude".to_string(),
            messages: vec![text_message(MessageRole::User, "hi")],
            max_output_tokens: Some(100),
            temperature: None,
            top_p: None,
            reasoning_effort: None,
            prompt_cache_ttl: None,
            stream: false,
            output_format: Some(OutputFormat {
                name: "test".to_string(),
                description: "test schema".to_string(),
                schema,
            }),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
        };
        let ant_req = AntRequest::try_from(request).unwrap();
        let format = ant_req.output_config.unwrap().format.unwrap();
        assert_eq!(
            format.schema.get("additionalProperties").unwrap().as_bool(),
            Some(false)
        );
    }

    #[test]
    fn ant_request_output_format_keeps_additional_properties_false() {
        let schema: schemars::Schema = serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "name": { "type": "string" }
            }
        })
        .try_into()
        .unwrap();
        let request = LLMRequest {
            api_type: ApiType::Anthropic,
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "claude".to_string(),
            messages: vec![text_message(MessageRole::User, "hi")],
            max_output_tokens: Some(100),
            temperature: None,
            top_p: None,
            reasoning_effort: None,
            prompt_cache_ttl: None,
            stream: false,
            output_format: Some(OutputFormat {
                name: "test".to_string(),
                description: "test schema".to_string(),
                schema,
            }),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
        };
        let ant_req = AntRequest::try_from(request).unwrap();
        let format = ant_req.output_config.unwrap().format.unwrap();
        assert_eq!(
            format.schema.get("additionalProperties").unwrap().as_bool(),
            Some(false)
        );
    }

    #[test]
    fn ant_request_effort_enables_adaptive_thinking() {
        let request = LLMRequest {
            api_type: ApiType::Anthropic,
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "claude".to_string(),
            messages: vec![text_message(MessageRole::User, "hi")],
            max_output_tokens: Some(100),
            temperature: None,
            top_p: None,
            reasoning_effort: Some(ReasoningEffort::Medium),
            prompt_cache_ttl: None,
            stream: false,
            output_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
        };
        let ant_req = AntRequest::try_from(request).unwrap();
        assert!(matches!(
            ant_req.thinking,
            Some(AntThinkingConfig::Adaptive(_))
        ));
    }

    #[test]
    fn ant_request_no_effort_disables_thinking() {
        let request = LLMRequest {
            api_type: ApiType::Anthropic,
            base_url: Some("https://api.anthropic.com/v1".to_string()),
            api_key: "key".to_string(),
            model: "claude".to_string(),
            messages: vec![text_message(MessageRole::User, "hi")],
            max_output_tokens: Some(100),
            temperature: None,
            top_p: None,
            reasoning_effort: None,
            prompt_cache_ttl: None,
            stream: false,
            output_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
        };
        let ant_req = AntRequest::try_from(request).unwrap();
        assert!(matches!(
            ant_req.thinking,
            Some(AntThinkingConfig::Disabled(_))
        ));
    }

    #[test]
    fn ant_response_conversion_assembles_assistant_message() {
        let response = AntResponse {
            id: "msg_123".to_string(),
            content: vec![AntMessagePart::Text(AntTextPart {
                part_type: "text".to_string(),
                text: "hello".to_string(),
            })],
            usage: AntUsage {
                input_tokens: 2,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
                output_tokens: 1,
            },
        };
        let llm_response = LLMResponse::from(response);
        assert_eq!(llm_response.id, "msg_123");
        assert_eq!(llm_response.created_at, None);
        assert_eq!(llm_response.message.role, MessageRole::Assistant);
        assert!(matches!(
            llm_response.message.content.first(),
            Some(MessagePart::Text(TextPart { text })) if text == "hello"
        ));
    }
}
