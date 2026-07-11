use base64::prelude::*;
use schemars::{JsonSchema, Schema, schema_for, schema_for_value};
use std::{fmt::Debug, fs, io, str::FromStr};

use erased_serde::Serialize as ErasedSerialize;
use serde::{Deserialize, Serialize};

use crate::errors::UnsupportedType;

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
    Function,
}

impl FromStr for MessageRole {
    type Err = UnsupportedType;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            "tool" => Ok(Self::Tool),
            "function" => Ok(Self::Function),
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

pub trait MessagePart {
    fn part_type(&self) -> MessagePartType;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextPart {
    pub text: String,
}

impl MessagePart for TextPart {
    fn part_type(&self) -> MessagePartType {
        MessagePartType::Text
    }
}

impl TextPart {
    pub fn new(text: String) -> Self {
        Self { text }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImagePart {
    pub data: String,
    pub is_base64: bool,
    pub mime_type: Option<String>,
}

impl MessagePart for ImagePart {
    fn part_type(&self) -> MessagePartType {
        MessagePartType::Image
    }
}

impl ImagePart {
    pub fn new(data: String, mime_type: String) -> Self {
        Self {
            data,
            is_base64: true,
            mime_type: Some(mime_type),
        }
    }

    pub fn try_from_file(file: String) -> Result<Self, io::Error> {
        let content = fs::read(file)?;
        let kind = file_format::FileFormat::from_bytes(&content);
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

#[derive(Debug, Serialize, Deserialize)]
pub struct AudioPart {
    data: String,
    mime_type: String,
}

impl MessagePart for AudioPart {
    fn part_type(&self) -> MessagePartType {
        MessagePartType::Audio
    }
}

impl AudioPart {
    pub fn new(data: String, mime_type: String) -> Self {
        Self { data, mime_type }
    }

    pub fn try_from_file(file: String) -> Result<Self, io::Error> {
        let content = fs::read(file)?;
        let kind = file_format::FileFormat::from_bytes(&content);
        let b64 = BASE64_STANDARD.encode(content);
        Ok(Self {
            data: b64,
            mime_type: kind.media_type().to_owned(),
        })
    }
}

pub trait Message: Debug + ErasedSerialize {
    fn role(&self) -> MessageRole;
    fn content(&self) -> Vec<Box<dyn MessagePart>>;
}

erased_serde::serialize_trait_object!(Message);

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Low,
    Medium,
    High,
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
            _ => Err(UnsupportedType {
                unsupported_type: s.to_owned(),
            }),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
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
    pub messages: Vec<Box<dyn Message>>,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub reasoning_effort: ReasoningEffort,
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
    messages: Vec<Box<dyn Message>>,
    max_output_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    reasoning_effort: ReasoningEffort,
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
            reasoning_effort: ReasoningEffort::default(),
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

    pub fn base_url(mut self, base_url: String) -> Self {
        self.base_url = Some(base_url);
        self
    }

    pub fn api_key(mut self, api_key: String) -> Self {
        self.api_key = api_key;
        self
    }

    #[must_use]
    pub fn model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    // To be continued...
}

pub struct LLM {}

impl LLM {
    pub async fn respond(&self, request: LLMRequest) {}
    pub async fn stream_response(&self, request: LLMRequest) {}
}
