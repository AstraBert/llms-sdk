use base64::prelude::*;
use futures_core::Stream;
use reqwest_retry::Jitter;
use schemars::{JsonSchema, Schema, schema_for, schema_for_value};
use std::collections::HashMap;
#[cfg(feature = "fs")]
use std::fs;
#[cfg(feature = "fs")]
use std::io;
#[cfg(feature = "fs")]
use std::path::PathBuf;
use std::{
    fmt::{Debug, Display},
    pin::Pin,
    str::FromStr,
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::errors::{InvalidInput, UnsupportedType};

/// Default base URL for the OpenAI API.
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
/// Default base URL for the Anthropic API.
pub const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
/// OpenAI chat completions API endpoint path.
pub const CHAT_COMPLETIONS_ENDPOINT: &str = "/chat/completions";
/// Anthropic messages API endpoint path.
pub const MESSAGES_ENDPOINT: &str = "/messages";
/// MIME types accepted for audio message parts.
pub const ALLOWED_AUDIO_TYPES: &[&str] = &[
    "audio/wav",
    "audio/mp3",
    "audio/mpeg",
    "audio/vnd.wav",
    "audio/vnd.wave",
];
/// MIME types accepted for document message parts.
pub const ALLOWED_DOCUMENT_TYPES: &[&str] = &["application/pdf", "text/plain"];
/// MIME types accepted for image message parts.
pub const ALLOWED_IMAGE_TYPES: &[&str] = &["image/png", "image/jpeg", "image/webp", "image/gif"];

/// Role of a participant in a chat message.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
}

impl FromStr for MessageRole {
    type Err = UnsupportedType;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            "tool" => Ok(Self::Tool),
            "system" => Ok(Self::System),
            _ => Err(UnsupportedType {
                unsupported_type: s.to_owned(),
            }),
        }
    }
}

/// A plain text content part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextPart {
    pub text: String,
}

impl TextPart {
    /// Creates a new text part from any string-like value.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// A reasoning/thinking content part, optionally signed by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThinkingPart {
    pub thinking: String,
    #[serde(default)]
    pub signature: Option<String>,
}

impl ThinkingPart {
    /// Creates a new thinking part without a signature.
    pub fn new(thinking: impl Into<String>) -> Self {
        Self {
            thinking: thinking.into(),
            signature: None,
        }
    }

    /// Creates a new thinking part with an associated signature.
    pub fn new_with_signature(thinking: impl Into<String>, signature: impl Into<String>) -> Self {
        Self {
            thinking: thinking.into(),
            signature: Some(signature.into()),
        }
    }
}

/// An image content part, either base64-encoded or loaded from a URL.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImagePart {
    pub data: String,
    pub is_base64: bool,
    pub mime_type: Option<String>,
}

impl ImagePart {
    /// Creates a new base64-encoded image part after validating its MIME type.
    pub fn new(
        data: impl Into<String>,
        mime_type: impl Into<String>,
    ) -> Result<Self, InvalidInput> {
        let media_type = mime_type.into();
        if !ALLOWED_IMAGE_TYPES.contains(&media_type.as_str()) {
            return Err(InvalidInput {
                reason: format!(
                    "Unsupported image type: {}. The supported image types are: {}",
                    media_type,
                    ALLOWED_IMAGE_TYPES.join(", ")
                ),
            });
        }
        Ok(Self {
            data: data.into(),
            is_base64: true,
            mime_type: Some(media_type),
        })
    }

    /// Creates an image part from raw bytes, inferring the MIME type.
    pub fn try_from_bytes(data: Vec<u8>) -> Result<Self, InvalidInput> {
        let kind = file_format::FileFormat::from_bytes(&data);
        if !ALLOWED_IMAGE_TYPES.contains(&kind.media_type()) {
            return Err(InvalidInput {
                reason: format!(
                    "Unsupported image type: {}. The supported image types are: {}",
                    kind.media_type(),
                    ALLOWED_IMAGE_TYPES.join(", ")
                ),
            });
        }
        let b64 = BASE64_STANDARD.encode(data);
        Ok(Self {
            data: b64,
            mime_type: Some(kind.media_type().to_owned()),
            is_base64: true,
        })
    }

    /// Reads an image file from disk and encodes it as a base64 image part.
    #[cfg(feature = "fs")]
    pub fn try_from_file(file: String) -> Result<Self, io::Error> {
        let content = fs::read(file)?;
        let kind = file_format::FileFormat::from_bytes(&content);
        if !ALLOWED_IMAGE_TYPES.contains(&kind.media_type()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Unsupported image type: {}. The supported image types are: {}",
                    kind.media_type(),
                    ALLOWED_IMAGE_TYPES.join(", ")
                ),
            ));
        }
        let b64 = BASE64_STANDARD.encode(content);
        Ok(Self {
            data: b64,
            mime_type: Some(kind.media_type().to_owned()),
            is_base64: true,
        })
    }

    /// Creates an image part that references an image URL.
    pub fn from_url(url: String) -> Self {
        Self {
            data: url,
            is_base64: false,
            mime_type: None,
        }
    }
}

/// An audio content part, base64-encoded.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioPart {
    pub data: String,
    pub mime_type: String,
}

impl AudioPart {
    /// Creates a new base64-encoded audio part after validating its MIME type.
    pub fn new(
        data: impl Into<String>,
        mime_type: impl Into<String>,
    ) -> Result<Self, InvalidInput> {
        let media_type = mime_type.into();
        if !ALLOWED_AUDIO_TYPES.contains(&media_type.as_str()) {
            return Err(InvalidInput {
                reason: format!(
                    "Unsupported audio type: {}. The supported audio types are: {}",
                    media_type,
                    ALLOWED_AUDIO_TYPES.join(", ")
                ),
            });
        }
        Ok(Self {
            data: data.into(),
            mime_type: media_type,
        })
    }

    /// Creates an audio part from raw bytes, inferring the MIME type.
    pub fn try_from_bytes(data: Vec<u8>) -> Result<Self, InvalidInput> {
        let kind = file_format::FileFormat::from_bytes(&data);
        if !ALLOWED_AUDIO_TYPES.contains(&kind.media_type()) {
            return Err(InvalidInput {
                reason: format!(
                    "Unsupported audio type: {}. The supported audio types are: {}",
                    kind.media_type(),
                    ALLOWED_AUDIO_TYPES.join(", ")
                ),
            });
        }
        let b64 = BASE64_STANDARD.encode(data);
        Ok(Self {
            data: b64,
            mime_type: kind.media_type().to_owned(),
        })
    }

    /// Reads an audio file from disk and encodes it as a base64 audio part.
    #[cfg(feature = "fs")]
    pub fn try_from_file(file: impl Into<PathBuf>) -> Result<Self, io::Error> {
        let content = fs::read(file.into())?;
        let kind = file_format::FileFormat::from_bytes(&content);
        if !ALLOWED_AUDIO_TYPES.contains(&kind.media_type()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Unsupported audio type: {}. The supported audio types are: {}",
                    kind.media_type(),
                    ALLOWED_AUDIO_TYPES.join(", ")
                ),
            ));
        }
        let b64 = BASE64_STANDARD.encode(content);
        Ok(Self {
            data: b64,
            mime_type: kind.media_type().to_owned(),
        })
    }
}

/// A document content part, either base64-encoded or loaded from a URL.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentPart {
    pub data: String,
    pub mime_type: Option<String>,
    pub is_base64: bool,
}

impl DocumentPart {
    /// Creates a new document part after validating its MIME type.
    pub fn new(
        data: impl Into<String>,
        mime_type: impl Into<String>,
    ) -> Result<Self, InvalidInput> {
        let media_type = mime_type.into();
        if !ALLOWED_DOCUMENT_TYPES.contains(&media_type.as_str()) {
            return Err(InvalidInput {
                reason: format!(
                    "Unsupported document type: {}. The supported document types are: {}",
                    media_type,
                    ALLOWED_DOCUMENT_TYPES.join(", ")
                ),
            });
        }
        Ok(Self {
            data: data.into(),
            is_base64: media_type == "application/pdf",
            mime_type: Some(media_type),
        })
    }

    /// Creates a base64-encoded PDF document part from raw bytes.
    pub fn try_from_pdf_bytes(data: Vec<u8>) -> Result<Self, InvalidInput> {
        let kind = file_format::FileFormat::from_bytes(&data);
        if kind.media_type() != "application/pdf" {
            return Err(InvalidInput {
                reason: format!(
                    "The bytes do not appear to belong to a PDF. Inferred file format: {}",
                    kind.media_type(),
                ),
            });
        }
        let b64 = BASE64_STANDARD.encode(data);
        Ok(Self {
            data: b64,
            mime_type: Some("application/pdf".to_string()),
            is_base64: true,
        })
    }

    /// Reads a plain text file from disk as a document part.
    #[cfg(feature = "fs")]
    pub fn try_from_text_file(file: String) -> Result<Self, io::Error> {
        let content = fs::read_to_string(file)?;
        Ok(Self {
            data: content,
            mime_type: Some("text/plain".to_string()),
            is_base64: false,
        })
    }

    /// Reads a PDF file from disk and encodes it as a base64 document part.
    #[cfg(feature = "fs")]
    pub fn try_from_pdf_file(file: String) -> Result<Self, io::Error> {
        let data = fs::read(file)?;
        let kind = file_format::FileFormat::from_bytes(&data);
        if kind.media_type() != "application/pdf" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "The bytes do not appear to belong to a PDF. Inferred file format: {}",
                    kind.media_type(),
                ),
            ));
        }
        let b64 = BASE64_STANDARD.encode(data);
        Ok(Self {
            data: b64,
            mime_type: Some("application/pdf".to_string()),
            is_base64: true,
        })
    }

    /// Creates a document part that references a document URL.
    pub fn from_url(url: String) -> Self {
        Self {
            data: url,
            mime_type: None,
            is_base64: false,
        }
    }
}

/// A tool/function call produced by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallPart {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// The result of a tool/function call returned to the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResultPart {
    pub tool_call_id: String,
    pub result: String,
}

/// A single content item inside a [`Message`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
#[non_exhaustive]
pub enum MessagePart {
    Text(TextPart),
    Image(ImagePart),
    Audio(AudioPart),
    Document(DocumentPart),
    Thinking(ThinkingPart),
    ToolCall(ToolCallPart),
    ToolResult(ToolResultPart),
}

/// A chat message with a role and one or more content parts.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: Vec<MessagePart>,
}

/// Supported LLM API provider.
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ApiType {
    OpenAI,
    Anthropic,
}

impl FromStr for ApiType {
    type Err = UnsupportedType;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(Self::OpenAI),
            "anthropic" => Ok(Self::Anthropic),
            _ => Err(UnsupportedType {
                unsupported_type: s.to_owned(),
            }),
        }
    }
}

impl Display for ApiType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenAI => write!(f, "openai"),
            Self::Anthropic => write!(f, "anthropic"),
        }
    }
}

impl ApiType {
    fn default_base_url(self) -> &'static str {
        match self {
            Self::Anthropic => DEFAULT_ANTHROPIC_BASE_URL,
            Self::OpenAI => DEFAULT_OPENAI_BASE_URL,
        }
    }
}

/// Amount of reasoning effort requested from the model.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ReasoningEffort {
    #[default]
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Maximum,
}

impl Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Minimal => write!(f, "minimal"),
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Maximum => write!(f, "maximum"),
            Self::Xhigh => write!(f, "xhigh"),
        }
    }
}

impl FromStr for ReasoningEffort {
    type Err = UnsupportedType;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Self::None),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "maximum" | "max" => Ok(Self::Maximum),
            _ => Err(UnsupportedType {
                unsupported_type: s.to_owned(),
            }),
        }
    }
}

/// Controls whether and how the model is allowed to call tools.
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ToolChoice {
    None,
    #[default]
    Auto,
    Required,
}

impl FromStr for ToolChoice {
    type Err = UnsupportedType;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Self::None),
            "auto" => Ok(Self::Auto),
            "required" => Ok(Self::Required),
            _ => Err(UnsupportedType {
                unsupported_type: s.to_owned(),
            }),
        }
    }
}

/// A tool definition exposed to the model.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Schema,
}

impl Tool {
    /// Creates a tool definition from a type that implements [`JsonSchema`].
    pub fn new<T: JsonSchema>(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: schema_for!(T),
        }
    }

    /// Creates a tool definition from an already-serializable parameters value.
    pub fn from_parameters_value(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: impl Serialize,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: schema_for_value!(parameters),
        }
    }
}

/// JSON schema describing a structured output format.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OutputFormat {
    pub name: String,
    pub description: String,
    pub schema: Schema,
}

/// A unified request to send to an LLM API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LLMRequest {
    /// Target API provider.
    pub api_type: ApiType,
    /// Custom base URL for the API. When `None`, the provider's default is used.
    pub base_url: Option<String>,
    /// API key used to authenticate the request.
    pub api_key: String,
    /// Model identifier to use for the request.
    pub model: String,
    /// Conversation history sent to the model.
    pub messages: Vec<Message>,
    /// Maximum number of tokens the model is allowed to generate.
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling parameter.
    pub top_p: Option<f32>,
    /// Level of reasoning effort requested from the model.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Prompt cache time-to-live hint (provider-specific format).
    pub prompt_cache_ttl: Option<String>,
    /// Whether to request a streamed response.
    pub stream: bool,
    /// Optional JSON schema for structured outputs.
    pub output_format: Option<OutputFormat>,
    /// Tool definitions made available to the model.
    pub tools: Option<Vec<Tool>>,
    /// Controls whether the model may call tools.
    pub tool_choice: Option<ToolChoice>,
    /// Whether the model may call multiple tools in parallel.
    pub parallel_tool_calls: bool,
}

/// Builder for constructing an [`LLMRequest`].
pub struct LLMRequestBuilder {
    api_type: ApiType,
    base_url: Option<String>,
    api_key: String,
    model: String,
    messages: Vec<Message>,
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    reasoning_effort: Option<ReasoningEffort>,
    prompt_cache_ttl: Option<String>,
    stream: bool,
    output_format: Option<OutputFormat>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    parallel_tool_calls: bool,
}

impl Default for LLMRequestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl LLMRequestBuilder {
    pub fn new() -> Self {
        Self {
            api_type: ApiType::OpenAI,
            base_url: None,
            api_key: String::new(),
            model: String::new(),
            messages: vec![],
            max_output_tokens: None,
            reasoning_effort: None,
            prompt_cache_ttl: None,
            stream: false,
            output_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: false,
            temperature: None,
            top_p: None,
        }
    }

    pub fn api_type(mut self, api_type: ApiType) -> Self {
        self.api_type = api_type;
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    #[must_use = "Must specify a model in order to use the LLM API"]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn add_tool(mut self, tool: Tool) -> Self {
        self.tools.get_or_insert_with(Vec::new).push(tool);
        self
    }

    #[must_use = "Must specify messages before sending a request"]
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    pub fn parallel_tool_calls(mut self, parallel_tool_calls: bool) -> Self {
        self.parallel_tool_calls = parallel_tool_calls;
        self
    }

    pub fn reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(reasoning_effort);
        self
    }

    pub fn output_format<T: JsonSchema>(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.output_format = Some(OutputFormat {
            name: name.into(),
            description: description.into(),
            schema: schema_for!(T),
        });
        self
    }

    pub fn output_format_from_schema(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        schema: Schema,
    ) -> Self {
        self.output_format = Some(OutputFormat {
            name: name.into(),
            description: description.into(),
            schema,
        });
        self
    }

    pub fn output_format_from_value(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        value: impl Serialize,
    ) -> Self {
        self.output_format = Some(OutputFormat {
            name: name.into(),
            description: description.into(),
            schema: schema_for_value!(value),
        });
        self
    }

    pub fn prompt_cache_ttl(mut self, ttl: impl Into<String>) -> Self {
        self.prompt_cache_ttl = Some(ttl.into());
        self
    }

    fn get_base_url(&self) -> String {
        if let Some(url) = &self.base_url {
            url.to_owned()
        } else {
            self.api_type.default_base_url().to_owned()
        }
    }

    pub fn build(self) -> LLMRequest {
        let base_url = self.get_base_url();
        LLMRequest {
            api_type: self.api_type,
            api_key: self.api_key,
            base_url: Some(base_url),
            model: self.model,
            stream: self.stream,
            reasoning_effort: self.reasoning_effort,
            temperature: self.temperature,
            top_p: self.top_p,
            max_output_tokens: self.max_output_tokens,
            tool_choice: self.tool_choice,
            tools: self.tools,
            parallel_tool_calls: self.parallel_tool_calls,
            prompt_cache_ttl: self.prompt_cache_ttl,
            output_format: self.output_format,
            messages: self.messages,
        }
    }
}

impl LLMRequest {
    pub(crate) fn base_url_or_default(&self) -> String {
        self.base_url
            .clone()
            .unwrap_or_else(|| self.api_type.default_base_url().to_owned())
    }

    pub fn builder() -> LLMRequestBuilder {
        LLMRequestBuilder::new()
    }
}

/// Configuration for transient request retries.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub min_retry_interval: Duration,
    pub max_retry_interval: Duration,
    pub jitter: Jitter,
    pub base: u32,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            min_retry_interval: Duration::from_millis(500_u64),
            max_retry_interval: Duration::from_millis(3000_u64),
            jitter: Jitter::Bounded,
            base: 2,
        }
    }
}

impl RetryPolicy {
    pub fn max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn min_retry_interval(mut self, min_retry_interval: Duration) -> Self {
        self.min_retry_interval = min_retry_interval;
        self
    }

    pub fn max_retry_interval(mut self, max_retry_interval: Duration) -> Self {
        self.max_retry_interval = max_retry_interval;
        self
    }

    pub fn jitter(mut self, jitter: Jitter) -> Self {
        self.jitter = jitter;
        self
    }

    pub fn base(mut self, base: u32) -> Self {
        self.base = base;
        self
    }
}

/// Token usage reported by the LLM API.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LLMUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: Option<u32>,
    pub cache_write_tokens: Option<u32>,
    pub other_tokens: Option<HashMap<String, u32>>,
}

/// A complete, non-streaming response from the LLM.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LLMResponse {
    /// Provider-generated response identifier.
    pub id: String,
    /// Unix timestamp of the response, when provided by the API.
    pub created_at: Option<u64>,
    /// The generated message.
    pub message: Message,
    /// Token usage reported for the request.
    pub usage: LLMUsage,
}

/// A partial text delta in a streaming response.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LLMStreamingDelta {
    /// Identifier of the response this delta belongs to.
    pub response_id: String,
    /// Unix timestamp of the response, when provided by the API.
    pub created_at: Option<u64>,
    /// Chunk of generated text, if any.
    pub delta: Option<String>,
    /// Whether this delta signals the end of the stream.
    pub stop: bool,
}

/// A partial reasoning/thinking delta in a streaming response.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LLMThinkingDelta {
    /// Identifier of the response this delta belongs to.
    pub response_id: String,
    /// Unix timestamp of the response, when provided by the API.
    pub created_at: Option<u64>,
    /// Chunk of reasoning text, if any.
    pub delta: Option<String>,
}

/// A partial tool call argument delta in a streaming response.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LLMToolDelta {
    /// Identifier of the response this delta belongs to.
    pub response_id: String,
    /// Identifier for the in-progress tool call.
    pub tool_call_id: String,
    /// Name of the tool being called.
    pub name: String,
    /// Partial JSON arguments accumulated so far.
    pub partial_arguments: String,
}

/// Final aggregated payload emitted at the end of a streaming response.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LLMStreamingComplete {
    /// Provider-generated response identifier.
    pub id: String,
    /// Unix timestamp of the response, when provided by the API.
    pub created_at: Option<u64>,
    /// The final assembled message.
    pub message: Message,
    /// All text deltas that make up the response.
    pub deltas: Vec<LLMStreamingDelta>,
    /// All reasoning deltas, if the model produced any.
    pub thinking_deltas: Option<Vec<LLMThinkingDelta>>,
    /// Token usage reported for the request, if provided.
    pub usage: Option<LLMUsage>,
    /// Complete tool calls parsed from the stream.
    pub tool_calls: Option<Vec<ToolCallPart>>,
}

/// A single item emitted by an [`LLMStream`].
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum LLMStreamingResponse {
    /// A chunk of generated text.
    Delta(LLMStreamingDelta),
    /// A chunk of tool call arguments.
    ToolDelta(LLMToolDelta),
    /// A chunk of reasoning text.
    ThinkingDelta(LLMThinkingDelta),
    /// The final aggregated response.
    Complete(LLMStreamingComplete),
}

pub type LLMStreamItem = Result<LLMStreamingResponse, Box<dyn std::error::Error + Send + Sync>>;
pub type LLMStream = Pin<Box<dyn Stream<Item = LLMStreamItem> + Send>>;

/// Checks whether a string is valid JSON.
///
/// Useful for validating accumulated streaming tool arguments before yielding
/// a complete [`ToolCallPart`].
pub fn is_valid_json(s: &str) -> bool {
    let v = serde_json::from_str::<serde_json::Value>(s);
    v.is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_valid_json_accepts_complete_object() {
        assert!(is_valid_json(r#"{"key": "value"}"#));
    }

    #[test]
    fn is_valid_json_rejects_incomplete_object() {
        assert!(!is_valid_json(r#"{"key": "value""#));
    }

    #[test]
    fn request_without_base_url_uses_the_provider_default() {
        let mut request = LLMRequest {
            api_type: ApiType::OpenAI,
            base_url: None,
            api_key: "key".to_string(),
            model: "model".to_string(),
            messages: vec![],
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

        assert_eq!(request.base_url_or_default(), DEFAULT_OPENAI_BASE_URL);
        request.api_type = ApiType::Anthropic;
        assert_eq!(request.base_url_or_default(), DEFAULT_ANTHROPIC_BASE_URL);
    }

    #[test]
    fn image_part_try_from_bytes_accepts_png() {
        let png = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let part = ImagePart::try_from_bytes(png.to_vec()).unwrap();
        assert_eq!(part.mime_type, Some("image/png".to_string()));
        assert!(part.is_base64);
    }

    #[test]
    fn image_part_try_from_bytes_rejects_unknown_format() {
        let part = ImagePart::try_from_bytes(vec![0x00, 0x01, 0x02, 0x03]);
        assert!(part.is_err());
    }

    #[test]
    fn audio_part_try_from_bytes_accepts_wav() {
        let wav = [
            0x52, 0x49, 0x46, 0x46, // RIFF
            0x00, 0x00, 0x00, 0x00, // size
            0x57, 0x41, 0x56, 0x45, // WAVE
        ];
        let part = AudioPart::try_from_bytes(wav.to_vec()).unwrap();
        assert!(ALLOWED_AUDIO_TYPES.contains(&part.mime_type.as_str()));
    }

    #[test]
    fn audio_part_try_from_bytes_accepts_mp3() {
        let mp3 = [
            0xFF, 0xFB, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        let part = AudioPart::try_from_bytes(mp3.to_vec()).unwrap();
        assert_eq!(part.mime_type, "audio/mpeg");
    }

    #[test]
    fn document_part_try_from_pdf_bytes_accepts_pdf() {
        let pdf = b"%PDF-1.4".to_vec();
        let part = DocumentPart::try_from_pdf_bytes(pdf).unwrap();
        assert_eq!(part.mime_type, Some("application/pdf".to_string()));
        assert!(part.is_base64);
    }

    #[test]
    fn document_part_try_from_pdf_bytes_rejects_non_pdf() {
        let part = DocumentPart::try_from_pdf_bytes(b"not a pdf".to_vec());
        assert!(part.is_err());
    }

    #[cfg(feature = "fs")]
    #[test]
    fn image_part_try_from_file_accepts_cat_jpeg() {
        let part = ImagePart::try_from_file("files/cat.jpeg".to_string()).unwrap();
        assert_eq!(part.mime_type, Some("image/jpeg".to_string()));
        assert!(part.is_base64);
    }

    #[cfg(feature = "fs")]
    #[test]
    fn audio_part_try_from_file_accepts_audio_wav() {
        let part = AudioPart::try_from_file("files/audio.wav".to_string()).unwrap();
        assert!(ALLOWED_AUDIO_TYPES.contains(&part.mime_type.as_str()));
    }

    #[cfg(feature = "fs")]
    #[test]
    fn audio_part_try_from_file_accepts_audio_mp3() {
        let part = AudioPart::try_from_file("files/audio.mp3".to_string()).unwrap();
        assert_eq!(part.mime_type, "audio/mpeg");
    }

    #[cfg(feature = "fs")]
    #[test]
    fn document_part_try_from_pdf_file_accepts_file_pdf() {
        let part = DocumentPart::try_from_pdf_file("files/file.pdf".to_string()).unwrap();
        assert_eq!(part.mime_type, Some("application/pdf".to_string()));
        assert!(part.is_base64);
    }
}
