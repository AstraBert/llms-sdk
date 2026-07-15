use std::str::FromStr;

use schemars::Schema;
use serde::{Deserialize, Serialize};

use crate::{
    ImagePart, MessageRole, ReasoningEffort, TextPart, ThinkingPart, Tool, ToolCallPart,
    ToolResultPart,
    errors::{InvalidTtl, UnsupportedType},
};

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
#[serde(untagged)]
pub enum AntMessagePart {
    Text(AntTextPart),
    Thinking(AntThinkingPart),
    Image(AntImagePart),
    ToolCall(AntToolCallPart),
    ToolResult(AntToolResultPart),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AntMessage {
    role: AntMessageRole,
    content: Vec<AntMessagePart>,
}

pub enum AllowedCacheTtl {
    FiveMinutes,
    ThirtyMinutes,
}

impl FromStr for AllowedCacheTtl {
    type Err = InvalidTtl;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "5m" => Ok(Self::FiveMinutes),
            "30m" => Ok(Self::ThirtyMinutes),
            _ => Err(InvalidTtl { ttl: s.to_string() }),
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

#[derive(Debug, Serialize, Deserialize, Clone)]
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
pub struct AntRequest {
    pub model: String,
    pub messages: Vec<AntMessage>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<AntOutputConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AntToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AntTool>>,
}
