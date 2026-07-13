use base64::prelude::*;
use reqwest_retry::Jitter;
use schemars::{JsonSchema, Schema, schema_for, schema_for_value};
#[cfg(feature = "fs")]
use std::fs;
#[cfg(feature = "fs")]
use std::path::PathBuf;
use std::{
    fmt::{Debug, Display},
    io,
    str::FromStr,
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::errors::{InvalidInput, UnsupportedType};

pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
pub const CHAT_COMPLETIONS_ENDPOINT: &str = "/chat/completions";
pub const MESSAGES_ENDPOINT: &str = "/messages";
pub const ALLOWED_AUDIO_TYPES: &[&str] = &["audio/wav", "audio/mp3"];
pub const ALLOWED_IMAGE_TYPES: &[&str] = &["image/png", "image/jpeg", "image/webp", "image/gif"];

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
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

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum MessagePartType {
    Text,
    Thinking,
    Image,
    Audio,
}

impl ToString for MessagePartType {
    fn to_string(&self) -> String {
        match self {
            Self::Text => "text".to_string(),
            Self::Image => "image".to_string(),
            Self::Audio => "audio".to_string(),
            Self::Thinking => "thinking".to_string(),
        }
    }
}

impl FromStr for MessagePartType {
    type Err = UnsupportedType;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "thinking" => Ok(Self::Thinking),
            "image" => Ok(Self::Image),
            "audio" => Ok(Self::Audio),
            _ => Err(UnsupportedType {
                unsupported_type: s.to_owned(),
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPart {
    pub text: String,
}

impl TextPart {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingPart {
    pub thinking: String,
}

impl ThinkingPart {
    pub fn new(thinking: impl Into<String>) -> Self {
        Self {
            thinking: thinking.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePart {
    pub data: String,
    pub is_base64: bool,
    pub mime_type: Option<String>,
}

impl ImagePart {
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

    pub fn from_url(url: String) -> Self {
        Self {
            data: url,
            is_base64: false,
            mime_type: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioPart {
    pub data: String,
    pub mime_type: String,
}

impl AudioPart {
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

    pub fn try_from_bytes(data: Vec<u8>) -> Result<Self, InvalidInput> {
        let kind = file_format::FileFormat::from_bytes(&data);
        if !ALLOWED_AUDIO_TYPES.contains(&&kind.media_type()) {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallPart {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultPart {
    pub tool_call_id: String,
    pub result: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum MessagePart {
    Text(TextPart),
    Image(ImagePart),
    Audio(AudioPart),
    Thinking(ThinkingPart),
    ToolCall(ToolCallPart),
    ToolResult(ToolResultPart),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: Vec<MessagePart>,
}

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

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Maximum,
}

impl Default for ReasoningEffort {
    fn default() -> Self {
        Self::Low
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

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    None,
    Auto,
    Required,
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Auto
    }
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Schema,
}

impl Tool {
    pub fn new<T: JsonSchema>(name: String, description: String) -> Self {
        Self {
            name,
            description,
            parameters: schema_for!(T),
        }
    }

    pub fn from_parameters_value(
        name: String,
        description: String,
        parameters: impl Serialize,
    ) -> Self {
        Self {
            name,
            description,
            parameters: schema_for_value!(parameters),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct LLMRequest {
    pub api_type: ApiType,
    pub base_url: Option<String>,
    pub api_key: String,
    pub model: String,
    pub messages: Vec<Message>,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub prompt_cache_ttl: Option<String>,
    pub stream: bool,
    pub output_format: Option<Schema>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: bool,
}

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
    output_format: Option<Schema>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    parallel_tool_calls: bool,
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

    #[must_use]
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

    pub fn output_format<T: JsonSchema>(mut self) -> Self {
        self.output_format = Some(schema_for!(T));
        self
    }

    pub fn output_format_from_schema(mut self, schema: Schema) -> Self {
        self.output_format = Some(schema);
        self
    }

    pub fn output_format_from_value(mut self, value: impl Serialize) -> Self {
        self.output_format = Some(schema_for_value!(value));
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
            match self.api_type {
                ApiType::Anthropic => DEFAULT_ANTHROPIC_BASE_URL.to_owned(),
                ApiType::OpenAI => DEFAULT_OPENAI_BASE_URL.to_owned(),
            }
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
    pub fn builder() -> LLMRequestBuilder {
        LLMRequestBuilder::new()
    }
}

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

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct LLMUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: u32,
    pub cache_write_tokens: Option<u32>,
    pub other_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LLMResponse {
    pub id: String,
    pub created_at: Option<u64>,
    pub messages: Vec<Message>,
    pub usage: LLMUsage,
}
