use serde::{Deserialize, Serialize};

use crate::{ImagePart, MessageRole, TextPart};

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
    #[serde(default)]
    pub cache_control: Option<CacheControl>,
}

impl From<TextPart> for AntTextPart {
    fn from(value: TextPart) -> Self {
        Self {
            part_type: "text".to_string(),
            text: value.text,
            cache_control: None,
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
    #[serde(default)]
    pub cache_control: Option<CacheControl>,
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
            cache_control: None,
        }
    }
}
