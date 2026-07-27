use base64::prelude::*;
use llms_sdk::ApiType as NativeApiType;
use llms_sdk::AudioPart as NativeAudioPart;
use llms_sdk::DocumentPart as NativeDocumentPart;
use llms_sdk::ImagePart as NativeImagePart;
use llms_sdk::LLMRequest as NativeLLMRequest;
use llms_sdk::LLMResponse as NativeLLMResponse;
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
use llms_sdk::{ALLOWED_AUDIO_TYPES, ALLOWED_IMAGE_TYPES};
use napi::bindgen_prelude::Either4;
use napi::bindgen_prelude::Either7;
use napi::bindgen_prelude::{Buffer, Either};
use schemars::Schema;
use std::collections::HashMap;
use std::fs;
use url::Url;

use napi_derive::napi;

/// Role of a message in the conversation.
#[napi(string_enum = "lowercase")]
#[derive(Debug, Clone, Copy)]
pub enum MessageRole {
  User,
  Assistant,
  System,
  Tool,
}

impl From<MessageRole> for NativeMessageRole {
  fn from(value: MessageRole) -> Self {
    match value {
      MessageRole::Assistant => Self::Assistant,
      MessageRole::System => Self::System,
      MessageRole::Tool => Self::Tool,
      MessageRole::User => Self::User,
    }
  }
}

impl From<NativeMessageRole> for MessageRole {
  fn from(value: NativeMessageRole) -> Self {
    match value {
      NativeMessageRole::Assistant => Self::Assistant,
      NativeMessageRole::System => Self::System,
      NativeMessageRole::Tool => Self::Tool,
      NativeMessageRole::User => Self::User,
    }
  }
}

/// Supported LLM API providers.
#[napi(string_enum = "lowercase")]
#[derive(Debug, Clone, Copy)]
pub enum ApiType {
  OpenAI,
  Anthropic,
}

impl From<ApiType> for NativeApiType {
  fn from(value: ApiType) -> Self {
    match value {
      ApiType::Anthropic => Self::Anthropic,
      ApiType::OpenAI => Self::OpenAI,
    }
  }
}

/// Amount of reasoning effort the model should expend.
#[napi(string_enum = "lowercase")]
#[derive(Debug, Clone, Copy)]
pub enum ReasoningEffort {
  None,
  Minimal,
  Low,
  Medium,
  High,
  Xhigh,
  Maximum,
}

impl From<ReasoningEffort> for NativeReasoningEffort {
  fn from(value: ReasoningEffort) -> Self {
    match value {
      ReasoningEffort::High => Self::High,
      ReasoningEffort::Low => Self::Low,
      ReasoningEffort::Maximum => Self::Maximum,
      ReasoningEffort::Medium => Self::Medium,
      ReasoningEffort::Minimal => Self::Minimal,
      ReasoningEffort::None => Self::None,
      ReasoningEffort::Xhigh => Self::Xhigh,
    }
  }
}

/// Controls whether the model is allowed to call tools.
#[napi(string_enum = "lowercase")]
#[derive(Debug, Clone, Copy)]
pub enum ToolChoice {
  None,
  Auto,
  Required,
}

impl From<ToolChoice> for NativeToolChoice {
  fn from(value: ToolChoice) -> Self {
    match value {
      ToolChoice::Auto => Self::Auto,
      ToolChoice::Required => Self::Required,
      ToolChoice::None => Self::None,
    }
  }
}

/// Plain-text segment of a message.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct TextPart {
  #[napi(js_name = "type", ts_type = "\"text\"")]
  pub r#type: String,
  pub text: String,
}

/// Image segment of a message.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ImagePart {
  #[napi(js_name = "type", ts_type = "\"image\"")]
  pub r#type: String,
  /// Raw image data (base64-encoded) or a URL.
  pub image_data: String,
  /// MIME type of the image, when known.
  pub mime_type: Option<String>,
  /// Whether `data` is base64-encoded (`true`) or a URL (`false`).
  pub is_base64: bool,
}

/// Audio segment of a message.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct AudioPart {
  #[napi(js_name = "type", ts_type = "\"audio\"")]
  pub r#type: String,
  /// Base64-encoded audio data.
  pub audio_data: String,
  /// MIME type of the audio (e.g. `audio/mpeg`).
  pub mime_type: String,
}

/// Document segment of a message (PDF or plain text).
#[napi(object)]
#[derive(Debug, Clone)]
pub struct DocumentPart {
  #[napi(js_name = "type", ts_type = "\"document\"")]
  pub r#type: String,
  /// Raw document data (base64-encoded) or a URL.
  pub document_data: String,
  /// MIME type of the document, when known.
  pub mime_type: Option<String>,
  /// Whether `data` is base64-encoded (`true`) or a URL (`false`).
  pub is_base64: bool,
}

/// A tool call issued by the model.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ToolCallPart {
  #[napi(js_name = "type", ts_type = "\"toolCall\"")]
  pub r#type: String,
  /// Unique identifier for this tool call.
  pub id: String,
  /// Name of the tool being invoked.
  pub name: String,
  /// JSON-encoded arguments for the tool call.
  pub arguments: String,
}

/// Result returned to the model after executing a tool call.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ToolResultPart {
  #[napi(js_name = "type", ts_type = "\"toolResult\"")]
  pub r#type: String,
  /// Identifier of the original tool call this result belongs to.
  pub tool_call_id: String,
  /// JSON-encoded result payload.
  pub result: String,
}

/// A reasoning/thinking block produced by the model.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct ThinkingPart {
  #[napi(js_name = "type", ts_type = "\"thinking\"")]
  pub r#type: String,
  /// The model's internal reasoning text.
  pub thinking: String,
  /// Cryptographic signature verifying the reasoning, when provided.
  pub signature: Option<String>,
}

/// A single item inside a [`Message`]'s content array.
#[napi(transparent)]
#[derive(Debug, Clone)]
pub struct MessagePart(
  pub  Either7<
    TextPart,
    AudioPart,
    DocumentPart,
    ImagePart,
    ToolCallPart,
    ToolResultPart,
    ThinkingPart,
  >,
);

impl From<MessagePart> for NativeMessagePart {
  fn from(value: MessagePart) -> Self {
    match value.0 {
      Either7::A(t) => NativeMessagePart::Text(NativeTextPart { text: t.text }),
      Either7::B(a) => NativeMessagePart::Audio(NativeAudioPart {
        data: a.audio_data,
        mime_type: a.mime_type,
      }),
      Either7::C(d) => NativeMessagePart::Document(NativeDocumentPart {
        data: d.document_data,
        mime_type: d.mime_type,
        is_base64: d.is_base64,
      }),
      Either7::D(i) => NativeMessagePart::Image(NativeImagePart {
        data: i.image_data,
        mime_type: i.mime_type,
        is_base64: i.is_base64,
      }),
      Either7::E(tc) => NativeMessagePart::ToolCall(NativeToolCallPart {
        id: tc.id,
        arguments: tc.arguments,
        name: tc.name,
      }),
      Either7::F(tr) => NativeMessagePart::ToolResult(NativeToolResultPart {
        tool_call_id: tr.tool_call_id,
        result: tr.result,
      }),
      Either7::G(t) => NativeMessagePart::Thinking(NativeThinkingPart {
        thinking: t.thinking,
        signature: t.signature,
      }),
    }
  }
}

impl From<NativeMessagePart> for MessagePart {
  fn from(value: NativeMessagePart) -> Self {
    match value {
      NativeMessagePart::Audio(a) => Self(Either7::B(AudioPart {
        r#type: "audio".to_string(),
        audio_data: a.data,
        mime_type: a.mime_type,
      })),
      NativeMessagePart::Document(d) => Self(Either7::C(DocumentPart {
        r#type: "document".to_string(),
        document_data: d.data,
        mime_type: d.mime_type,
        is_base64: d.is_base64,
      })),
      NativeMessagePart::Image(i) => Self(Either7::D(ImagePart {
        r#type: "image".to_string(),
        image_data: i.data,
        mime_type: i.mime_type,
        is_base64: i.is_base64,
      })),
      NativeMessagePart::Text(t) => Self(Either7::A(TextPart {
        text: t.text,
        r#type: "text".to_string(),
      })),
      NativeMessagePart::Thinking(t) => Self(Either7::G(ThinkingPart {
        r#type: "thinking".to_string(),
        thinking: t.thinking,
        signature: t.signature,
      })),
      NativeMessagePart::ToolCall(tc) => Self(Either7::E(ToolCallPart {
        r#type: "toolCall".to_string(),
        id: tc.id,
        name: tc.name,
        arguments: tc.arguments,
      })),
      NativeMessagePart::ToolResult(tr) => Self(Either7::F(ToolResultPart {
        r#type: "toolResult".to_string(),
        tool_call_id: tr.tool_call_id,
        result: tr.result,
      })),
      _ => unreachable!("Should not reach this arm"),
    }
  }
}

/// Create an `ImagePart` from a URL, file path, or raw bytes.
///
/// @param input - Either a URL/path (`string`) or a `Buffer` containing image data.
/// @returns An `ImagePart` ready to be added to a `Message`.
#[napi]
pub fn image_part(input: Either<String, Buffer>) -> napi::Result<ImagePart> {
  match input {
    Either::A(s) => match Url::parse(&s) {
      Ok(u) => {
        if matches!(u.scheme(), "http" | "https") {
          Ok(ImagePart {
            r#type: "image".to_string(),
            image_data: s,
            mime_type: None,
            is_base64: false,
          })
        } else {
          let data = fs::read(&s)?;
          let format = file_format::FileFormat::from_bytes(&data);
          if !ALLOWED_IMAGE_TYPES.contains(&format.media_type()) {
            return Err(napi::Error::new(
              napi::Status::InvalidArg,
              format!(
                "Unsupported image type: {}. The supported image types are: {}",
                format.media_type(),
                ALLOWED_IMAGE_TYPES.join(", ")
              ),
            ));
          }
          Ok(ImagePart {
            r#type: "image".to_string(),
            image_data: BASE64_STANDARD.encode(&data),
            mime_type: Some(format.media_type().to_owned()),
            is_base64: true,
          })
        }
      }
      Err(_) => {
        let data = fs::read(&s)?;
        let format = file_format::FileFormat::from_bytes(&data);
        if !ALLOWED_IMAGE_TYPES.contains(&format.media_type()) {
          return Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!(
              "Unsupported image type: {}. The supported image types are: {}",
              format.media_type(),
              ALLOWED_IMAGE_TYPES.join(", ")
            ),
          ));
        }
        Ok(ImagePart {
          r#type: "image".to_string(),
          image_data: BASE64_STANDARD.encode(&data),
          mime_type: Some(format.media_type().to_owned()),
          is_base64: true,
        })
      }
    },
    Either::B(buf) => {
      let data: Vec<u8> = buf.into();
      let format = file_format::FileFormat::from_bytes(&data);
      if !ALLOWED_IMAGE_TYPES.contains(&format.media_type()) {
        return Err(napi::Error::new(
          napi::Status::InvalidArg,
          format!(
            "Unsupported image type: {}. The supported image types are: {}",
            format.media_type(),
            ALLOWED_IMAGE_TYPES.join(", ")
          ),
        ));
      }
      Ok(ImagePart {
        r#type: "image".to_string(),
        image_data: BASE64_STANDARD.encode(&data),
        mime_type: Some(format.media_type().to_owned()),
        is_base64: true,
      })
    }
  }
}

/// Create a `DocumentPart` from a URL, file path, or raw bytes.
///
/// @param input - Either a URL/path (`string`) or a `Buffer` containing PDF data.
/// @returns A `DocumentPart` ready to be added to a `Message`.
#[napi]
pub fn document_part(input: Either<String, Buffer>) -> napi::Result<DocumentPart> {
  match input {
    Either::A(s) => match Url::parse(&s) {
      Ok(u) => {
        if matches!(u.scheme(), "http" | "https") {
          Ok(DocumentPart {
            r#type: "document".to_string(),
            document_data: s,
            mime_type: None,
            is_base64: false,
          })
        } else {
          let format = file_format::FileFormat::from_file(&s).map_err(|e| {
            napi::Error::new(
              napi::Status::InvalidArg,
              format!("Could not infer format from file: {}", e),
            )
          })?;
          if format.media_type() == "application/pdf" {
            let data = fs::read(&s)?;
            Ok(DocumentPart {
              r#type: "document".to_string(),
              document_data: BASE64_STANDARD.encode(data),
              mime_type: Some("application/pdf".to_string()),
              is_base64: true,
            })
          } else if format.media_type().starts_with("text/") {
            let data = fs::read_to_string(&s)?;
            Ok(DocumentPart {
              r#type: "document".to_string(),
              document_data: data,
              mime_type: Some("text/plain".to_string()),
              is_base64: false,
            })
          } else {
            Err(napi::Error::new(
              napi::Status::InvalidArg,
              format!(
                "Expected either a PDF or a text file, found media type: {}",
                format.media_type()
              ),
            ))
          }
        }
      }
      Err(_) => {
        let format = file_format::FileFormat::from_file(&s).map_err(|e| {
          napi::Error::new(
            napi::Status::InvalidArg,
            format!("Could not infer format from file: {}", e),
          )
        })?;
        if format.media_type() == "application/pdf" {
          let data = fs::read(&s)?;
          Ok(DocumentPart {
            r#type: "document".to_string(),
            document_data: BASE64_STANDARD.encode(data),
            mime_type: Some("application/pdf".to_string()),
            is_base64: true,
          })
        } else if format.media_type().starts_with("text/") {
          let data = fs::read_to_string(&s)?;
          Ok(DocumentPart {
            r#type: "document".to_string(),
            document_data: data,
            mime_type: Some("text/plain".to_string()),
            is_base64: false,
          })
        } else {
          Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!(
              "Expected either a PDF or a text file, found media type: {}",
              format.media_type()
            ),
          ))
        }
      }
    },
    Either::B(buf) => {
      let data: Vec<u8> = buf.into();
      let format = file_format::FileFormat::from_bytes(&data);
      if format.media_type() != "application/pdf" {
        return Err(napi::Error::new(
          napi::Status::InvalidArg,
          format!(
            "Input file should be a PDF, found media type: {}",
            format.media_type()
          ),
        ));
      }
      Ok(DocumentPart {
        r#type: "document".to_string(),
        document_data: BASE64_STANDARD.encode(&data),
        mime_type: Some(format.media_type().to_owned()),
        is_base64: true,
      })
    }
  }
}

/// Create an `AudioPart` from a file path or raw bytes.
///
/// @param input - Either a file path (`string`) or a `Buffer` containing audio data.
/// @returns An `AudioPart` ready to be added to a `Message`.
#[napi]
pub fn audio_part(input: Either<String, Buffer>) -> napi::Result<AudioPart> {
  let data: Vec<u8> = match input {
    Either::A(s) => fs::read(&s)?,
    Either::B(buf) => buf.into(),
  };
  let format = file_format::FileFormat::from_bytes(&data);
  if !ALLOWED_AUDIO_TYPES.contains(&format.media_type()) {
    return Err(napi::Error::new(
      napi::Status::InvalidArg,
      format!(
        "Unsupported audio type: {}. The supported audio types are: {}",
        format.media_type(),
        ALLOWED_AUDIO_TYPES.join(", ")
      ),
    ));
  }
  Ok(AudioPart {
    r#type: "audio".to_string(),
    audio_data: BASE64_STANDARD.encode(&data),
    mime_type: format.media_type().to_owned(),
  })
}

/// A single turn in the conversation.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Message {
  /// Who produced this message (user, assistant, system, or tool).
  pub role: MessageRole,
  /// Ordered list of content parts that make up the message.
  pub content: Vec<MessagePart>,
}

impl From<Message> for NativeMessage {
  fn from(value: Message) -> Self {
    let mut content: Vec<NativeMessagePart> = vec![];
    for p in value.content {
      content.push(p.into());
    }
    Self {
      role: value.role.into(),
      content,
    }
  }
}

impl From<NativeMessage> for Message {
  fn from(value: NativeMessage) -> Self {
    let mut content: Vec<MessagePart> = vec![];
    for p in value.content {
      content.push(p.into());
    }
    Self {
      content,
      role: value.role.into(),
    }
  }
}

/// A tool definition made available to the model.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct Tool {
  /// Name of the tool.
  pub name: String,
  /// Human-readable description of what the tool does.
  pub description: String,
  /// JSON schema representing the tool parameters.
  pub parameters: serde_json::Value,
}

impl TryFrom<Tool> for NativeTool {
  type Error = napi::Error;
  fn try_from(value: Tool) -> Result<Self, Self::Error> {
    let parameters: Schema = value.parameters.try_into()?;
    Ok(Self {
      name: value.name,
      description: value.description,
      parameters,
    })
  }
}

/// Structured-output format enforced on the model response.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct OutputFormat {
  /// Name of the output format.
  pub name: String,
  /// Human-readable description.
  pub description: String,
  /// JSON schema the model output must conform to.
  pub schema: serde_json::Value,
}

impl TryFrom<OutputFormat> for NativeOutputFormat {
  type Error = napi::Error;
  fn try_from(value: OutputFormat) -> Result<Self, Self::Error> {
    let schema: Schema = value.schema.try_into()?;
    Ok(Self {
      name: value.name,
      description: value.description,
      schema,
    })
  }
}

/// Request payload sent to the LLM.
#[napi(object)]
#[derive(Debug, Clone)]
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
  /// Sampling temperature (0 = deterministic, higher = more random).
  pub temperature: Option<f64>,
  /// Nucleus sampling parameter (0–1).
  pub top_p: Option<f64>,
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

impl TryFrom<LLMRequest> for NativeLLMRequest {
  type Error = napi::Error;
  fn try_from(value: LLMRequest) -> Result<Self, Self::Error> {
    let mut tools: Option<Vec<NativeTool>> = None;
    if let Some(ts) = value.tools {
      for t in ts {
        tools.get_or_insert_with(Vec::new).push(t.try_into()?);
      }
    }
    let mut output_format: Option<NativeOutputFormat> = None;
    if let Some(of) = value.output_format {
      output_format = Some(of.try_into()?)
    }
    let mut messages: Vec<NativeMessage> = vec![];
    for m in value.messages {
      messages.push(m.into());
    }
    Ok(Self {
      tools,
      output_format,
      messages,
      max_output_tokens: value.max_output_tokens,
      temperature: value.temperature.map(|t| t as f32),
      top_p: value.top_p.map(|t| t as f32),
      tool_choice: value.tool_choice.map(|tc| tc.into()),
      parallel_tool_calls: value.parallel_tool_calls,
      prompt_cache_ttl: value.prompt_cache_ttl,
      api_key: value.api_key,
      api_type: value.api_type.into(),
      stream: value.stream,
      reasoning_effort: value.reasoning_effort.map(|r| r.into()),
      model: value.model,
      base_url: value.base_url,
    })
  }
}

/// Token usage reported by the LLM API.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct LLMUsage {
  /// Tokens consumed by the prompt.
  pub input_tokens: u32,
  /// Tokens generated in the response.
  pub output_tokens: u32,
  /// Tokens read from a provider cache, when applicable.
  pub cache_read_tokens: Option<u32>,
  /// Tokens written to a provider cache, when applicable.
  pub cache_write_tokens: Option<u32>,
  /// Any additional tokens counted by the provider.
  pub other_tokens: Option<HashMap<String, u32>>,
}

/// A complete, non-streaming response from the LLM.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct LLMResponse {
  /// Provider-generated response identifier.
  pub id: String,
  /// Unix timestamp of the response, when provided by the API.
  pub created_at: Option<u32>,
  /// The generated message.
  pub message: Message,
  /// Token usage reported for the request.
  pub usage: LLMUsage,
}

impl From<NativeLLMResponse> for LLMResponse {
  fn from(value: NativeLLMResponse) -> Self {
    Self {
      id: value.id,
      created_at: value.created_at.map(|v| v as u32),
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
#[napi(object)]
#[derive(Debug, Clone)]
pub struct LLMStreamingDelta {
  #[napi(js_name = "type", ts_type = "\"textDelta\"")]
  pub r#type: String,
  /// Identifier of the response this delta belongs to.
  pub response_id: String,
  /// Unix timestamp of the response, when provided by the API.
  pub created_at: Option<u32>,
  /// Chunk of generated text, if any.
  pub text_delta: Option<String>,
  /// Whether this delta signals the end of the stream.
  pub stop: bool,
}

impl From<NativeLLMStreamingDelta> for LLMStreamingDelta {
  fn from(value: NativeLLMStreamingDelta) -> Self {
    Self {
      r#type: "textDelta".to_string(),
      response_id: value.response_id,
      created_at: value.created_at.map(|c| c as u32),
      text_delta: value.delta,
      stop: value.stop,
    }
  }
}

/// A partial reasoning/thinking delta in a streaming response.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct LLMThinkingDelta {
  #[napi(js_name = "type", ts_type = "\"thinkingDelta\"")]
  pub r#type: String,
  /// Identifier of the response this delta belongs to.
  pub response_id: String,
  /// Unix timestamp of the response, when provided by the API.
  pub created_at: Option<u32>,
  /// Chunk of reasoning text, if any.
  pub thinking_delta: Option<String>,
}

impl From<NativeLLMThinkingDelta> for LLMThinkingDelta {
  fn from(value: NativeLLMThinkingDelta) -> Self {
    Self {
      r#type: "thinkingDelta".to_string(),
      response_id: value.response_id,
      created_at: value.created_at.map(|c| c as u32),
      thinking_delta: value.delta,
    }
  }
}

/// A partial tool call argument delta in a streaming response.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct LLMToolDelta {
  #[napi(js_name = "type", ts_type = "\"toolDelta\"")]
  pub r#type: String,
  /// Identifier for the in-progress tool call.
  pub tool_call_id: String,
  /// Name of the tool being called.
  pub name: String,
  /// Partial JSON arguments accumulated so far.
  pub partial_arguments: String,
}

/// Final aggregated payload emitted at the end of a streaming response.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct LLMStreamingComplete {
  #[napi(js_name = "type", ts_type = "\"complete\"")]
  pub r#type: String,
  /// Provider-generated response identifier.
  pub id: String,
  /// Unix timestamp of the response, when provided by the API.
  pub created_at: Option<u32>,
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

/// A single item emitted by a streaming LLM response.
#[napi(transparent)]
#[derive(Debug, Clone)]
pub struct LLMStreamingResponse(
  pub Either4<LLMStreamingDelta, LLMToolDelta, LLMThinkingDelta, LLMStreamingComplete>,
);

impl From<NativeLLMStreamingResponse> for LLMStreamingResponse {
  fn from(value: NativeLLMStreamingResponse) -> Self {
    match value {
      NativeLLMStreamingResponse::Complete(c) => {
        let mut deltas: Vec<LLMStreamingDelta> = vec![];
        for d in c.deltas {
          deltas.push(d.into());
        }
        let mut thinking_deltas: Option<Vec<LLMThinkingDelta>> = None;
        if let Some(td) = c.thinking_deltas {
          for t in td {
            thinking_deltas.get_or_insert_with(Vec::new).push(t.into());
          }
        }
        let mut tool_calls: Option<Vec<ToolCallPart>> = None;
        if let Some(tc) = c.tool_calls {
          for t in tc {
            tool_calls.get_or_insert_with(Vec::new).push(ToolCallPart {
              r#type: "toolCall".to_string(),
              id: t.id,
              name: t.name,
              arguments: t.arguments,
            });
          }
        }
        Self(Either4::D(LLMStreamingComplete {
          r#type: "complete".to_string(),
          id: c.id,
          created_at: c.created_at.map(|c| c as u32),
          message: c.message.into(),
          deltas,
          thinking_deltas,
          usage: c.usage.map(|u| LLMUsage {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_read_tokens: u.cache_read_tokens,
            cache_write_tokens: u.cache_write_tokens,
            other_tokens: u.other_tokens,
          }),
          tool_calls,
        }))
      }
      NativeLLMStreamingResponse::Delta(d) => Self(Either4::A(d.into())),
      NativeLLMStreamingResponse::ThinkingDelta(td) => Self(Either4::C(td.into())),
      NativeLLMStreamingResponse::ToolDelta(td) => Self(Either4::B(LLMToolDelta {
        r#type: "toolDelta".to_string(),
        tool_call_id: td.tool_call_id,
        name: td.name,
        partial_arguments: td.partial_arguments,
      })),
    }
  }
}
