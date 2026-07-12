use reqwest_middleware::ClientBuilder;
use reqwest_retry::{RetryTransientMiddleware, policies::ExponentialBackoff};
use schemars::Schema;
use serde::{Deserialize, Serialize};

use crate::{
    ApiType, AudioPart, CHAT_COMPLETIONS_ENDPOINT, ImagePart, LLMRequest, LLMResponse, LLMUsage,
    Message, MessagePart, MessageRole, ReasoningEffort, RetryPolicy, TextPart, Tool, ToolChoice,
    errors::{StreamParamError, UnsupportedPartType},
};

#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAIClient {
    pub retry_policy: RetryPolicy,
}

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

impl TryFrom<Message> for OpenAIMessage {
    type Error = UnsupportedPartType;

    fn try_from(value: Message) -> Result<Self, Self::Error> {
        let mut content = vec![];
        for m in value.content {
            content.push(OpenAIMessagePart::try_from(m)?);
        }
        Ok(Self {
            role: OpenAIMessageRole::from(value.role),
            content,
        })
    }
}

impl Into<Message> for OpenAIMessage {
    fn into(self) -> Message {
        Message {
            role: {
                match self.role {
                    OpenAIMessageRole::Assistant => MessageRole::Assistant,
                    OpenAIMessageRole::Developer => MessageRole::System,
                    OpenAIMessageRole::Function => MessageRole::Function,
                    OpenAIMessageRole::User => MessageRole::User,
                    OpenAIMessageRole::Tool => MessageRole::Tool,
                }
            },
            content: {
                let mut cs = vec![];
                for c in self.content {
                    let part = match c {
                        OpenAIMessagePart::Audio(a) => MessagePart::Audio(AudioPart {
                            data: a.input_audio.data,
                            mime_type: a.input_audio.format,
                        }),
                        OpenAIMessagePart::Text(t) => MessagePart::Text(TextPart { text: t.text }),
                        OpenAIMessagePart::Image(i) => {
                            let mime_type = if i.image_url.starts_with("data:") {
                                let (d, _) =
                                    i.image_url.split_once(";").expect("Malformed base64 data");
                                Some(d.replace("data:", "").trim().to_owned())
                            } else {
                                None
                            };
                            MessagePart::Image(ImagePart {
                                data: i.image_url.clone(),
                                is_base64: i.image_url.starts_with("data:"),
                                mime_type,
                            })
                        }
                    };
                    cs.push(part);
                }
                cs
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum PromptCacheMode {
    Implicit,
    Explicit,
}

impl Default for PromptCacheMode {
    fn default() -> Self {
        Self::Implicit
    }
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
}

impl From<ReasoningEffort> for OpenAIReasoningEffort {
    fn from(value: ReasoningEffort) -> Self {
        match value {
            ReasoningEffort::None => Self::None,
            ReasoningEffort::Minimal => Self::Minimal,
            ReasoningEffort::Low => Self::Low,
            ReasoningEffort::High => Self::High,
            ReasoningEffort::Medium => Self::Medium,
            ReasoningEffort::Maximum => Self::Xhigh,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormatType {
    JsonSchema,
}

impl Default for ResponseFormatType {
    fn default() -> Self {
        Self::JsonSchema
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: ResponseFormatType,
    json_schema: Schema,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_options: Option<PromptCacheOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<OpenAIReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    stream: bool,
    parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

impl TryFrom<LLMRequest> for OpenAIRequest {
    type Error = UnsupportedPartType;
    fn try_from(value: LLMRequest) -> Result<Self, Self::Error> {
        let mut messages = vec![];
        for m in value.messages {
            messages.push(OpenAIMessage::try_from(m)?);
        }
        Ok(Self {
            model: value.model,
            messages,
            max_completion_tokens: value.max_output_tokens,
            temperature: value.temperature,
            tool_choice: value.tool_choice,
            tools: value.tools,
            prompt_cache_options: value.prompt_cache_ttl.map(|ttl| PromptCacheOptions {
                mode: PromptCacheMode::default(),
                ttl,
            }),
            parallel_tool_calls: value.parallel_tool_calls,
            stream: value.stream,
            top_p: value.top_p,
            response_format: value.output_format.map(|f| ResponseFormat {
                format_type: ResponseFormatType::default(),
                json_schema: f,
            }),
            reasoning_effort: value
                .reasoning_effort
                .map(|r| OpenAIReasoningEffort::from(r)),
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
            cache_read_tokens: value.prompt_tokens_details.cached_tokens,
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
pub struct ChatCompletion {
    index: usize,
    message: OpenAIMessage,
    finish_reason: String,
}

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
        let messages: Vec<Message> = value
            .choices
            .first()
            .map_or(vec![], |c| vec![c.message.to_owned().into()]);
        Self {
            messages,
            usage: LLMUsage::from(value.usage),
            id: value.id,
            created_at: Some(value.created),
        }
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
            .body(body)
            .send()
            .await?
            .json::<OpenAIResponse>()
            .await?;

        Ok(LLMResponse::from(response))
    }
}
