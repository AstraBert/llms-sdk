use base64::prelude::*;
use llms_sdk::AudioPart as NativeAudioPart;
use llms_sdk::DocumentPart as NativeDocumentPart;
use llms_sdk::ImagePart as NativeImagePart;
use llms_sdk::MessagePart as NativeMessagePart;
use llms_sdk::TextPart as NativeTextPart;
use llms_sdk::ThinkingPart as NativeThinkingPart;
use llms_sdk::ToolCallPart as NativeToolCallPart;
use llms_sdk::ToolResultPart as NativeToolResultPart;
use llms_sdk::{ALLOWED_AUDIO_TYPES, ALLOWED_IMAGE_TYPES};
use napi::bindgen_prelude::Buffer;
use std::fs;

use napi_derive::napi;

#[napi]
pub enum MessageRole {
  User,
  Assistant,
  System,
  Tool,
}

#[napi(object)]
pub struct TextPart {
  pub text: String,
}

#[napi(object)]
pub struct ImagePart {
  pub data: String,
  pub mime_type: Option<String>,
  pub is_base64: bool,
}

#[napi]
pub fn image_part_from_file(file: String) -> napi::Result<ImagePart> {
  let data = fs::read(file)?;
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
    data: BASE64_STANDARD.encode(&data),
    mime_type: Some(format.media_type().to_owned()),
    is_base64: true,
  })
}

#[napi]
pub fn image_part_from_buffer(buffer: Buffer) -> napi::Result<ImagePart> {
  let data: Vec<u8> = buffer.into();
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
    data: BASE64_STANDARD.encode(&data),
    mime_type: Some(format.media_type().to_owned()),
    is_base64: true,
  })
}

#[napi]
pub fn image_part_from_url(url: String) -> ImagePart {
  ImagePart {
    data: url,
    mime_type: None,
    is_base64: false,
  }
}

#[napi(object)]
pub struct AudioPart {
  pub data: String,
  pub mime_type: String,
}

#[napi]
pub fn audio_part_from_file(file: String) -> napi::Result<AudioPart> {
  let data = fs::read(file)?;
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
    data: BASE64_STANDARD.encode(&data),
    mime_type: format.media_type().to_owned(),
  })
}

#[napi]
pub fn audio_part_from_buffer(buffer: Buffer) -> napi::Result<AudioPart> {
  let data: Vec<u8> = buffer.into();
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
    data: BASE64_STANDARD.encode(&data),
    mime_type: format.media_type().to_owned(),
  })
}

#[napi(object)]
pub struct DocumentPart {
  pub data: String,
  pub mime_type: Option<String>,
  pub is_base64: bool,
}

#[napi]
pub fn document_part_from_pdf(file: String) -> napi::Result<DocumentPart> {
  let data = fs::read(file)?;
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
    data: BASE64_STANDARD.encode(&data),
    mime_type: Some(format.media_type().to_owned()),
    is_base64: true,
  })
}

#[napi]
pub fn document_part_from_pdf_buffer(buffer: Buffer) -> napi::Result<DocumentPart> {
  let data: Vec<u8> = buffer.into();
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
    data: BASE64_STANDARD.encode(&data),
    mime_type: Some(format.media_type().to_owned()),
    is_base64: true,
  })
}

#[napi]
pub fn document_part_from_txt(file: String) -> napi::Result<DocumentPart> {
  let data = fs::read(file)?;
  let format = file_format::FileFormat::from_bytes(&data);
  if !format.media_type().starts_with("text/") {
    return Err(napi::Error::new(
      napi::Status::InvalidArg,
      format!(
        "Input file should be a plain-text file, found media type: {}",
        format.media_type()
      ),
    ));
  }
  Ok(DocumentPart {
    data: BASE64_STANDARD.encode(&data),
    mime_type: Some("text/plain".to_string()),
    is_base64: true,
  })
}

#[napi(object)]
pub struct ToolCallPart {
  pub id: String,
  pub name: String,
  pub arguments: String,
}

#[napi(object)]
pub struct ToolResultPart {
  pub tool_call_id: String,
  pub result: String,
}

#[napi(object)]
pub struct ThinkingPart {
  pub thinking: String,
  pub signature: Option<String>,
}

#[napi]
pub enum MessagePart {
  Text(TextPart),
  Image(ImagePart),
  Document(DocumentPart),
  Audio(AudioPart),
  ToolCall(ToolCallPart),
  ToolResult(ToolResultPart),
  Thinking(ThinkingPart),
}

impl From<MessagePart> for NativeMessagePart {
  fn from(value: MessagePart) -> Self {
    match value {
      MessagePart::Text(t) => NativeMessagePart::Text(NativeTextPart { text: t.text }),
      MessagePart::Audio(a) => NativeMessagePart::Audio(NativeAudioPart {
        data: a.data,
        mime_type: a.mime_type,
      }),
      MessagePart::Document(d) => NativeMessagePart::Document(NativeDocumentPart {
        data: d.data,
        mime_type: d.mime_type,
        is_base64: d.is_base64,
      }),
      MessagePart::Image(i) => NativeMessagePart::Image(NativeImagePart {
        data: i.data,
        mime_type: i.mime_type,
        is_base64: i.is_base64,
      }),
      MessagePart::ToolCall(tc) => NativeMessagePart::ToolCall(NativeToolCallPart {
        id: tc.id,
        arguments: tc.arguments,
        name: tc.name,
      }),
      MessagePart::ToolResult(tr) => NativeMessagePart::ToolResult(NativeToolResultPart {
        tool_call_id: tr.tool_call_id,
        result: tr.result,
      }),
      MessagePart::Thinking(t) => NativeMessagePart::Thinking(NativeThinkingPart {
        thinking: t.thinking,
        signature: t.signature,
      }),
    }
  }
}
