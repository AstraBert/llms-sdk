use std::collections::HashMap;

use llms_sdk::ApiType as NativeApiType;
use llms_sdk::AudioPart as NativeAudioPart;
use llms_sdk::DocumentPart as NativeDocumentPart;
use llms_sdk::ImagePart as NativeImagePart;
use llms_sdk::LLMRequest as NativeLLMRequest;
use llms_sdk::LLMResponse as NativeLLMResponse;
use llms_sdk::LLMStreamingComplete as NativeLLMStreamingComplete;
use llms_sdk::LLMStreamingDelta as NativeLLMStreamingDelta;
use llms_sdk::LLMStreamingResponse as NativeLLMStreamingResponse;
use llms_sdk::LLMThinkingDelta as NativeLLMThinkingDelta;
use llms_sdk::Message as NativeMessage;
use llms_sdk::MessagePart as NativeMessagePart;
use llms_sdk::MessageRole as NativeMessageRole;
use llms_sdk::OutputFormat as NativeOutputFormat;
use llms_sdk::ReasoningEffort as NativeReasoningEffort;
use llms_sdk::TextPart as NativeTextPart;
use llms_sdk::ThinkingPart as NativeThinkingPart;
use llms_sdk::Tool as NativeTool;
use llms_sdk::ToolCallPart as NativeToolCallPart;
use llms_sdk::ToolChoice as NativeToolChoice;
use llms_sdk::ToolResultPart as NativeToolResultPart;
use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

/// A plain-text segment of a [`Message`].
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct TextPart {
    text: String,
}

/// A reasoning / thinking segment produced by models that support chain-of-thought output.
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ThinkingPart {
    thinking: String,
    signature: Option<String>,
}

/// An image segment of a [`Message`], either as a base-64 string or a URL.
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ImagePart {
    image_data: String,
    is_base64: bool,
    mime_type: Option<String>,
}

/// Raw binary data used when constructing media parts from buffers instead of URLs.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct BufferData {
    #[serde(with = "serde_bytes")]
    #[tsify(type = "Uint8Array")]
    bytes: Vec<u8>,
}

/// Input type accepted by helper constructors: either a remote URL or raw [`BufferData`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(untagged)]
pub enum UrlOrBuffer {
    Url(String),
    Buffer(BufferData),
}

/// Build an [`ImagePart`] from a URL or raw bytes.
#[wasm_bindgen(js_name = imagePart)]
pub fn image_part(input: UrlOrBuffer) -> Result<ImagePart, JsError> {
    let native_part = match input {
        UrlOrBuffer::Url(u) => NativeImagePart::from_url(u),
        UrlOrBuffer::Buffer(b) => {
            NativeImagePart::try_from_bytes(b.bytes).map_err(|e| JsError::new(&e.to_string()))?
        }
    };
    Ok(ImagePart {
        image_data: native_part.data,
        is_base64: native_part.is_base64,
        mime_type: native_part.mime_type,
    })
}

/// A document segment of a [`Message`], typically a PDF.
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct DocumentPart {
    document_data: String,
    is_base64: bool,
    mime_type: Option<String>,
}

/// Build a [`DocumentPart`] from a URL or raw PDF bytes.
#[wasm_bindgen(js_name = documentPart)]
pub fn document_part(input: UrlOrBuffer) -> Result<DocumentPart, JsError> {
    let native_part = match input {
        UrlOrBuffer::Url(u) => NativeDocumentPart::from_url(u),
        UrlOrBuffer::Buffer(b) => NativeDocumentPart::try_from_pdf_bytes(b.bytes)
            .map_err(|e| JsError::new(&e.to_string()))?,
    };
    Ok(DocumentPart {
        document_data: native_part.data,
        is_base64: native_part.is_base64,
        mime_type: native_part.mime_type,
    })
}

/// An audio segment of a [`Message`].
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct AudioPart {
    audio_data: String,
    mime_type: String,
}

/// Build an [`AudioPart`] from raw audio bytes.
#[wasm_bindgen(js_name = audioPart)]
pub fn audio_part(input: BufferData) -> Result<AudioPart, JsError> {
    let native_part =
        NativeAudioPart::try_from_bytes(input.bytes).map_err(|e| JsError::new(&e.to_string()))?;
    Ok(AudioPart {
        audio_data: native_part.data,
        mime_type: native_part.mime_type,
    })
}

/// A request from the model to invoke a tool.
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ToolCallPart {
    id: String,
    name: String,
    arguments: String,
}

/// The result returned to the model after a tool was executed.
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ToolResultPart {
    tool_call_id: String,
    result: String,
}

/// One of the possible content pieces inside a [`Message`].
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(tag = "type")]
pub enum MessagePart {
    #[serde(rename = "text")]
    Text(TextPart),
    #[serde(rename = "audio")]
    Audio(AudioPart),
    #[serde(rename = "document")]
    Document(DocumentPart),
    #[serde(rename = "image")]
    Image(ImagePart),
    #[serde(rename = "toolCall")]
    ToolCall(ToolCallPart),
    #[serde(rename = "toolResult")]
    ToolResult(ToolResultPart),
    #[serde(rename = "thinking")]
    Thinking(ThinkingPart),
}

/// The role of a participant in a conversation turn.
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    Tool,
    System,
}

/// A single turn in a chat conversation.
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct Message {
    role: MessageRole,
    content: Vec<MessagePart>,
}

/// Build a text message, with an optional role (defaults to user if left undefined).
#[wasm_bindgen(js_name = textMessage)]
pub fn text_message(content: String, role: Option<MessageRole>) -> Message {
    Message {
        role: role.unwrap_or(MessageRole::User),
        content: vec![MessagePart::Text(TextPart { text: content })],
    }
}

impl From<MessagePart> for NativeMessagePart {
    fn from(value: MessagePart) -> Self {
        match value {
            MessagePart::Audio(a) => NativeMessagePart::Audio(NativeAudioPart {
                data: a.audio_data,
                mime_type: a.mime_type,
            }),
            MessagePart::Document(d) => NativeMessagePart::Document(NativeDocumentPart {
                data: d.document_data,
                mime_type: d.mime_type,
                is_base64: d.is_base64,
            }),
            MessagePart::Image(i) => NativeMessagePart::Image(NativeImagePart {
                data: i.image_data,
                mime_type: i.mime_type,
                is_base64: i.is_base64,
            }),
            MessagePart::Text(t) => NativeMessagePart::Text(NativeTextPart { text: t.text }),
            MessagePart::Thinking(t) => NativeMessagePart::Thinking(NativeThinkingPart {
                thinking: t.thinking,
                signature: t.signature,
            }),
            MessagePart::ToolCall(tc) => NativeMessagePart::ToolCall(NativeToolCallPart {
                id: tc.id,
                name: tc.name,
                arguments: tc.arguments,
            }),
            MessagePart::ToolResult(tr) => NativeMessagePart::ToolResult(NativeToolResultPart {
                tool_call_id: tr.tool_call_id,
                result: tr.result,
            }),
        }
    }
}

impl From<NativeMessagePart> for MessagePart {
    fn from(value: NativeMessagePart) -> Self {
        match value {
            NativeMessagePart::Audio(a) => MessagePart::Audio(AudioPart {
                audio_data: a.data,
                mime_type: a.mime_type,
            }),
            NativeMessagePart::Document(d) => MessagePart::Document(DocumentPart {
                document_data: d.data,
                is_base64: d.is_base64,
                mime_type: d.mime_type,
            }),
            NativeMessagePart::Image(i) => MessagePart::Image(ImagePart {
                image_data: i.data,
                is_base64: i.is_base64,
                mime_type: i.mime_type,
            }),
            NativeMessagePart::Text(t) => MessagePart::Text(TextPart { text: t.text }),
            NativeMessagePart::Thinking(t) => MessagePart::Thinking(ThinkingPart {
                thinking: t.thinking,
                signature: t.signature,
            }),
            NativeMessagePart::ToolCall(tc) => MessagePart::ToolCall(ToolCallPart {
                id: tc.id,
                name: tc.name,
                arguments: tc.arguments,
            }),
            NativeMessagePart::ToolResult(tr) => MessagePart::ToolResult(ToolResultPart {
                tool_call_id: tr.tool_call_id,
                result: tr.result,
            }),
            _ => unreachable!("Unsupported part type"),
        }
    }
}

impl From<Message> for NativeMessage {
    fn from(value: Message) -> Self {
        let role: NativeMessageRole = match value.role {
            MessageRole::Assistant => NativeMessageRole::Assistant,
            MessageRole::System => NativeMessageRole::System,
            MessageRole::Tool => NativeMessageRole::Tool,
            MessageRole::User => NativeMessageRole::User,
        };
        let mut content: Vec<NativeMessagePart> = vec![];
        for part in value.content {
            content.push(part.into());
        }
        Self { role, content }
    }
}

impl From<NativeMessage> for Message {
    fn from(value: NativeMessage) -> Self {
        let role = match value.role {
            NativeMessageRole::Assistant => MessageRole::Assistant,
            NativeMessageRole::System => MessageRole::System,
            NativeMessageRole::Tool => MessageRole::Tool,
            NativeMessageRole::User => MessageRole::User,
        };
        let mut content: Vec<MessagePart> = vec![];
        for part in value.content {
            content.push(part.into());
        }
        Self { role, content }
    }
}

/// A tool definition exposed to the model.
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct Tool {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl TryFrom<Tool> for NativeTool {
    type Error = JsError;
    fn try_from(value: Tool) -> Result<Self, Self::Error> {
        Ok(Self {
            name: value.name,
            description: value.description,
            parameters: value
                .parameters
                .try_into()
                .map_err(|e: serde_json::Error| JsError::new(&e.to_string()))?,
        })
    }
}

/// A structured output format requested from the model (e.g. JSON Schema).
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct OutputFormat {
    name: String,
    description: String,
    schema: serde_json::Value,
}

/// Supported LLM provider APIs.
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum ApiType {
    Anthropic,
    OpenAI,
}

/// Controls whether and how the model may call tools.
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
}

/// How much reasoning effort the model should expend.
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Maximum,
}

impl TryFrom<OutputFormat> for NativeOutputFormat {
    type Error = JsError;
    fn try_from(value: OutputFormat) -> Result<Self, Self::Error> {
        Ok(Self {
            name: value.name,
            description: value.description,
            schema: value
                .schema
                .try_into()
                .map_err(|e: serde_json::Error| JsError::new(&e.to_string()))?,
        })
    }
}

/// Full request payload sent to the LLM.
#[derive(Debug, Serialize, Deserialize, Tsify, Clone)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct LLMRequest {
    api_type: ApiType,
    #[tsify(optional)]
    base_url: Option<String>,
    api_key: String,
    model: String,
    messages: Vec<Message>,
    #[tsify(optional)]
    max_output_tokens: Option<u32>,
    #[tsify(optional)]
    temperature: Option<f32>,
    #[tsify(optional)]
    top_p: Option<f32>,
    #[tsify(optional)]
    reasoning_effort: Option<ReasoningEffort>,
    #[tsify(optional)]
    prompt_cache_ttl: Option<String>,
    stream: bool,
    #[tsify(optional)]
    output_format: Option<OutputFormat>,
    #[tsify(optional)]
    tools: Option<Vec<Tool>>,
    #[tsify(optional)]
    tool_choice: Option<ToolChoice>,
    parallel_tool_calls: bool,
}

impl TryFrom<LLMRequest> for NativeLLMRequest {
    type Error = JsError;
    fn try_from(value: LLMRequest) -> Result<Self, Self::Error> {
        let mut messages: Vec<NativeMessage> = vec![];
        for m in value.messages {
            messages.push(m.into());
        }
        let tool_choice: Option<NativeToolChoice> = value.tool_choice.map(|t| match t {
            ToolChoice::Auto => NativeToolChoice::Auto,
            ToolChoice::None => NativeToolChoice::None,
            ToolChoice::Required => NativeToolChoice::Required,
        });
        let reasoning_effort: Option<NativeReasoningEffort> =
            value.reasoning_effort.map(|r| match r {
                ReasoningEffort::High => NativeReasoningEffort::High,
                ReasoningEffort::Low => NativeReasoningEffort::Low,
                ReasoningEffort::Maximum => NativeReasoningEffort::Maximum,
                ReasoningEffort::Medium => NativeReasoningEffort::Medium,
                ReasoningEffort::Minimal => NativeReasoningEffort::Minimal,
                ReasoningEffort::None => NativeReasoningEffort::None,
                ReasoningEffort::Xhigh => NativeReasoningEffort::Xhigh,
            });
        let api_type: NativeApiType = match value.api_type {
            ApiType::Anthropic => NativeApiType::Anthropic,
            ApiType::OpenAI => NativeApiType::OpenAI,
        };
        let output_format: Option<NativeOutputFormat> = match value.output_format {
            Some(v) => Some(NativeOutputFormat::try_from(v)?),
            None => None,
        };
        let mut tools: Option<Vec<NativeTool>> = None;
        if let Some(ts) = value.tools {
            for t in ts {
                tools.get_or_insert_with(Vec::new).push(t.try_into()?);
            }
        }
        Ok(Self {
            api_type,
            messages,
            model: value.model,
            tool_choice,
            tools,
            parallel_tool_calls: value.parallel_tool_calls,
            max_output_tokens: value.max_output_tokens,
            prompt_cache_ttl: value.prompt_cache_ttl,
            top_p: value.top_p,
            base_url: value.base_url,
            api_key: value.api_key,
            temperature: value.temperature,
            reasoning_effort,
            stream: value.stream,
            output_format,
        })
    }
}

/// Token usage statistics returned by the provider.
#[derive(Debug, Serialize, Deserialize, Clone, Tsify)]
#[tsify(from_wasm_abi, into_wasm_abi)]
pub struct LLMUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: Option<u32>,
    pub cache_write_tokens: Option<u32>,
    #[tsify(optional)]
    pub other_tokens: Option<HashMap<String, u32>>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct LLMResponse {
    /// Provider-generated response identifier.
    pub id: String,
    /// Unix timestamp of the response, when provided by the API.
    #[tsify(optional)]
    pub created_at: Option<u64>,
    /// The generated message.
    pub message: Message,
    /// Token usage reported for the request.
    pub usage: LLMUsage,
}

impl From<NativeLLMResponse> for LLMResponse {
    fn from(value: NativeLLMResponse) -> Self {
        Self {
            id: value.id,
            created_at: value.created_at,
            message: value.message.into(),
            usage: LLMUsage {
                input_tokens: value.usage.input_tokens,
                output_tokens: value.usage.output_tokens,
                cache_read_tokens: value.usage.cache_read_tokens,
                cache_write_tokens: value.usage.cache_write_tokens,
                other_tokens: value.usage.other_tokens,
            },
        }
    }
}

/// A partial text delta in a streaming response.
#[derive(Debug, Serialize, Deserialize, Clone, Tsify)]
#[tsify(from_wasm_abi, into_wasm_abi)]
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
#[derive(Debug, Serialize, Deserialize, Clone, Tsify)]
#[tsify(from_wasm_abi, into_wasm_abi)]
pub struct LLMThinkingDelta {
    /// Identifier of the response this delta belongs to.
    pub response_id: String,
    /// Unix timestamp of the response, when provided by the API.
    pub created_at: Option<u64>,
    /// Chunk of reasoning text, if any.
    pub delta: Option<String>,
}

impl From<NativeLLMStreamingDelta> for LLMStreamingDelta {
    fn from(value: NativeLLMStreamingDelta) -> Self {
        Self {
            delta: value.delta,
            created_at: value.created_at,
            stop: value.stop,
            response_id: value.response_id,
        }
    }
}

impl From<NativeLLMThinkingDelta> for LLMThinkingDelta {
    fn from(value: NativeLLMThinkingDelta) -> Self {
        Self {
            delta: value.delta,
            created_at: value.created_at,
            response_id: value.response_id,
        }
    }
}

/// A partial tool call argument delta in a streaming response.
#[derive(Debug, Serialize, Deserialize, Clone, Tsify)]
#[tsify(from_wasm_abi, into_wasm_abi)]
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
#[derive(Debug, Serialize, Deserialize, Clone, Tsify)]
#[tsify(from_wasm_abi, into_wasm_abi)]
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

impl From<NativeLLMStreamingComplete> for LLMStreamingComplete {
    fn from(value: NativeLLMStreamingComplete) -> Self {
        let mut deltas: Vec<LLMStreamingDelta> = vec![];
        for d in value.deltas {
            deltas.push(d.into());
        }
        let mut thinking_deltas: Option<Vec<LLMThinkingDelta>> = None;
        if let Some(tds) = value.thinking_deltas {
            for t in tds {
                thinking_deltas.get_or_insert_with(Vec::new).push(t.into())
            }
        }
        let mut tool_calls: Option<Vec<ToolCallPart>> = None;
        if let Some(tcs) = value.tool_calls {
            for t in tcs {
                tool_calls.get_or_insert_with(Vec::new).push(ToolCallPart {
                    id: t.id,
                    name: t.name,
                    arguments: t.arguments,
                });
            }
        }
        Self {
            created_at: value.created_at,
            usage: value.usage.map(|u| LLMUsage {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
                cache_read_tokens: u.cache_read_tokens,
                cache_write_tokens: u.cache_write_tokens,
                other_tokens: u.other_tokens,
            }),
            id: value.id,
            message: value.message.into(),
            deltas,
            thinking_deltas,
            tool_calls,
        }
    }
}

/// A single item emitted by an [`LLMStream`].
#[derive(Debug, Serialize, Deserialize, Clone, Tsify)]
#[tsify(from_wasm_abi, into_wasm_abi)]
#[serde(tag = "type")]
pub enum LLMStreamingResponse {
    /// A chunk of generated text.
    #[serde(rename = "delta")]
    Delta(LLMStreamingDelta),
    /// A chunk of tool call arguments.
    #[serde(rename = "toolDelta")]
    ToolDelta(LLMToolDelta),
    /// A chunk of reasoning text.
    #[serde(rename = "thinkingDelta")]
    ThinkingDelta(LLMThinkingDelta),
    /// The final aggregated response.
    #[serde(rename = "complete")]
    Complete(LLMStreamingComplete),
}

impl From<NativeLLMStreamingResponse> for LLMStreamingResponse {
    fn from(value: NativeLLMStreamingResponse) -> Self {
        match value {
            NativeLLMStreamingResponse::Complete(c) => LLMStreamingResponse::Complete(c.into()),
            NativeLLMStreamingResponse::Delta(d) => LLMStreamingResponse::Delta(d.into()),
            NativeLLMStreamingResponse::ThinkingDelta(td) => {
                LLMStreamingResponse::ThinkingDelta(td.into())
            }
            NativeLLMStreamingResponse::ToolDelta(td) => {
                LLMStreamingResponse::ToolDelta(LLMToolDelta {
                    response_id: td.response_id,
                    tool_call_id: td.tool_call_id,
                    name: td.name,
                    partial_arguments: td.partial_arguments,
                })
            }
        }
    }
}
