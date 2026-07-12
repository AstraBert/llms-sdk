use serde::{Deserialize, Serialize};

use crate::{
    ApiType, AudioPart, ImagePart, MessagePart, MessageRole, TextPart, errors::UnsupportedPartType,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAIClient {}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIMessageRole {
    Developer,
    User,
    Assistant,
    Tool,
    Function,
}

impl From<MessageRole> for OpenAIMessageRole {
    fn from(value: MessageRole) -> Self {
        match value {
            MessageRole::Assistant => Self::Assistant,
            MessageRole::Function => Self::Function,
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
pub struct OpenAIImagePart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub image_url: String,
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
            image_url,
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
                format: value.mime_type,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIMessagePart {
    Text(OpenAITextPart),
    Audio(OpenAIAudioPart),
    Image(OpenAIImagePart),
}

impl TryFrom<MessagePart> for OpenAIMessagePart {
    type Error = UnsupportedPartType;
    fn try_from(value: MessagePart) -> Result<Self, Self::Error> {
        match value {
            MessagePart::Text(t) => Ok(Self::Text(OpenAITextPart::from(t))),
            MessagePart::Audio(a) => Ok(Self::Audio(OpenAIAudioPart::from(a))),
            MessagePart::Image(i) => Ok(Self::Image(OpenAIImagePart::from(i))),
            MessagePart::Thinking(_) => Err(UnsupportedPartType {
                part_type: "thinking".to_string(),
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
pub struct OpenAIMessage {
    pub role: OpenAIMessageRole,
    pub content: Vec<OpenAIMessagePart>,
}

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
    // To be continued...
}
