use std::str::FromStr;

use reqwest_middleware::ClientBuilder;
use reqwest_retry::{RetryTransientMiddleware, policies::ExponentialBackoff};
use schemars::Schema;
use serde::{Deserialize, Serialize};

use crate::{
    ApiType, DocumentPart, ImagePart, LLMRequest, LLMResponse, LLMUsage, MESSAGES_ENDPOINT,
    Message, MessagePart, MessageRole, ReasoningEffort, RetryPolicy, TextPart, ThinkingPart, Tool,
    ToolCallPart, ToolChoice, ToolResultPart,
    errors::{
        InvalidAntRequestConversion, InvalidTtl, StreamParamError, UnsupportedPartType,
        UnsupportedType,
    },
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
    pub signature: String,
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

impl Into<MessagePart> for AntMessagePart {
    fn into(self) -> MessagePart {
        match self {
            Self::Text(t) => MessagePart::Text(TextPart { text: t.text }),
            Self::Document(d) => {
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
            Self::Image(i) => {
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
            Self::Thinking(t) => MessagePart::Thinking(ThinkingPart {
                thinking: t.thinking,
                signature: t.signature,
            }),
            Self::ToolCall(tc) => MessagePart::ToolCall(ToolCallPart {
                id: tc.id,
                name: tc.name,
                arguments: serde_json::to_string(&tc.input)
                    .expect("Tool input should be serializable"),
            }),
            Self::ToolResult(tr) => MessagePart::ToolResult(ToolResultPart {
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

impl ToString for AllowedCacheTtl {
    fn to_string(&self) -> String {
        match self {
            Self::FiveMinutes => "5m".to_string(),
            Self::OneHour => "1h".to_string(),
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
    json_schema: Schema,
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
            Some(ts.iter().cloned().map(|t| AntTool::from(t)).collect())
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
            json_schema: f.schema,
        });
        let output_config = if output_format.is_some() || effort.is_some() {
            Some(AntOutputConfig {
                effort: effort,
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
                    display: "summary".to_string(),
                })
            },
        );
        let top_p = if let Some(tp) = value.top_p {
            if let Some(_) = value.temperature {
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

#[derive(Debug, Clone, Copy, Default)]
pub struct AntClient {
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
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await?
            .error_for_status()?
            .json::<AntResponse>()
            .await?;

        Ok(LLMResponse::from(response))
    }
}
