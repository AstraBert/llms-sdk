use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod llms_sdk {
    use pyo3_async_runtimes::tokio::future_into_py;
    use std::collections::HashMap;
    use std::fs;

    use base64::prelude::*;
    use either::Either;
    use llms_sdk::ApiType as NativeApiType;
    use llms_sdk::DEFAULT_OPENAI_BASE_URL;
    use llms_sdk::LLM as NativeLLM;
    use llms_sdk::LLMRequest as NativeLLMRequest;
    use llms_sdk::LLMResponse as NativeLLMResponse;
    use llms_sdk::LLMUsage as NativeLLMUsage;
    use llms_sdk::Message as NativeMessage;
    use llms_sdk::MessagePart as NativeMessagePart;
    use llms_sdk::MessageRole as NativeMessageRole;
    use llms_sdk::OutputFormat as NativeOutputFormat;
    use llms_sdk::ReasoningEffort as NativeReasoningEffort;
    use llms_sdk::RetryPolicy as NativeRetryPolicy;
    use llms_sdk::Tool as NativeTool;
    use llms_sdk::ToolChoice as NativeToolChoice;
    use llms_sdk::{
        ALLOWED_IMAGE_TYPES, AudioPart, DocumentPart, ImagePart, TextPart, ThinkingPart,
        ToolCallPart, ToolResultPart,
    };
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::types::PyList;
    use pyo3::types::PyType;
    use pyo3::{
        exceptions::{PyAttributeError, PyKeyError, PyValueError},
        prelude::*,
        types::PyDict,
    };
    use pythonize::depythonize;
    use pythonize::pythonize;
    use schemars::Schema;
    use serde_json::Value;
    use std::time::Duration;
    use url::Url;

    #[pyclass(eq, hash, frozen, from_py_object)]
    #[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
    pub enum PartType {
        Audio,
        Document,
        Image,
        Text,
        Thinking,
        ToolCall,
        ToolResult,
    }

    #[pymethods]
    impl PartType {
        #[new]
        fn new(part_type: String) -> PyResult<Self> {
            match part_type.as_str() {
                "text" => Ok(Self::Text),
                "audio" => Ok(Self::Audio),
                "image" => Ok(Self::Image),
                "thinking" => Ok(Self::Thinking),
                "document" => Ok(Self::Document),
                "tool_result" => Ok(Self::ToolResult),
                "tool_call" => Ok(Self::ToolCall),
                _ => Err(PyValueError::new_err(format!(
                    "Invalid part type: {}",
                    part_type
                ))),
            }
        }

        fn __repr__(&self) -> String {
            match *self {
                Self::Text => "text".to_string(),
                Self::Image => "image".to_string(),
                Self::Audio => "audio".to_string(),
                Self::ToolCall => "tool_call".to_string(),
                Self::ToolResult => "tool_result".to_string(),
                Self::Thinking => "thinking".to_string(),
                Self::Document => "document".to_string(),
            }
        }

        fn __str__(&self) -> String {
            match *self {
                Self::Text => "text".to_string(),
                Self::Image => "image".to_string(),
                Self::Audio => "audio".to_string(),
                Self::ToolCall => "tool_call".to_string(),
                Self::ToolResult => "tool_result".to_string(),
                Self::Thinking => "thinking".to_string(),
                Self::Document => "document".to_string(),
            }
        }
    }

    #[pyclass(frozen)]
    #[derive(Debug, FromPyObject)]
    #[allow(clippy::enum_variant_names)]
    pub enum MessagePart {
        TextPart {
            #[pyo3(attribute("type"), default = PartType::Text)]
            r#type: PartType,
            text: String,
        },
        ImagePart {
            #[pyo3(attribute("type"), default = PartType::Image)]
            r#type: PartType,
            image_data: String,
            is_base64: bool,
            mime_type: Option<String>,
        },
        AudioPart {
            #[pyo3(attribute("type"), default = PartType::Audio)]
            r#type: PartType,
            audio_data: String,
            mime_type: String,
        },
        DocumentPart {
            #[pyo3(attribute("type"), default = PartType::Document)]
            r#type: PartType,
            document_data: String,
            is_base64: bool,
            mime_type: Option<String>,
        },
        ThinkingPart {
            #[pyo3(attribute("type"), default = PartType::Thinking)]
            r#type: PartType,
            thinking: String,
            signature: Option<String>,
        },
        ToolCallPart {
            #[pyo3(attribute("type"), default = PartType::ToolCall)]
            r#type: PartType,
            id: String,
            name: String,
            arguments: String,
        },
        ToolResultPart {
            #[pyo3(attribute("type"), default = PartType::ToolResult)]
            r#type: PartType,
            tool_call_id: String,
            result: String,
        },
    }

    impl MessagePart {
        fn as_clone(&self) -> Self {
            match self {
                Self::TextPart { text, r#type: tp } => Self::TextPart {
                    r#type: tp.to_owned(),
                    text: text.clone(),
                },
                Self::AudioPart {
                    audio_data,
                    mime_type,
                    r#type: tp,
                } => Self::AudioPart {
                    r#type: tp.to_owned(),
                    audio_data: audio_data.clone(),
                    mime_type: mime_type.clone(),
                },
                Self::ImagePart {
                    image_data,
                    mime_type,
                    r#type: tp,
                    is_base64,
                } => Self::ImagePart {
                    r#type: tp.to_owned(),
                    image_data: image_data.to_owned(),
                    is_base64: *is_base64,
                    mime_type: mime_type.to_owned(),
                },
                Self::DocumentPart {
                    document_data,
                    mime_type,
                    r#type: tp,
                    is_base64,
                } => Self::DocumentPart {
                    r#type: tp.to_owned(),
                    document_data: document_data.to_owned(),
                    is_base64: *is_base64,
                    mime_type: mime_type.to_owned(),
                },
                Self::ToolCallPart {
                    id,
                    name,
                    arguments,
                    r#type: tp,
                } => Self::ToolCallPart {
                    r#type: tp.to_owned(),
                    id: id.clone(),
                    name: name.to_owned(),
                    arguments: arguments.to_owned(),
                },
                Self::ToolResultPart {
                    tool_call_id,
                    result,
                    r#type: tp,
                } => Self::ToolResultPart {
                    r#type: tp.to_owned(),
                    tool_call_id: tool_call_id.clone(),
                    result: result.clone(),
                },
                Self::ThinkingPart {
                    thinking,
                    signature,
                    r#type: tp,
                } => Self::ThinkingPart {
                    r#type: tp.to_owned(),
                    thinking: thinking.clone(),
                    signature: signature.clone(),
                },
            }
        }
    }

    #[pymethods]
    impl MessagePart {
        #[new]
        #[pyo3(signature = (part_type, **kwargs))]
        fn new(part_type: String, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
            if let Some(args) = kwargs {
                let tp = PartType::new(part_type)?;
                match tp {
                    PartType::Text => {
                        let text_res = args.get_item("text")?;
                        match text_res {
                            Some(t) => Ok(Self::TextPart {
                                r#type: tp,
                                text: t.to_string(),
                            }),
                            None => Err(PyKeyError::new_err(
                                "Key 'text' not found among kwargs passed to constructor, but type was set to 'text'",
                            )),
                        }
                    }
                    PartType::Audio => {
                        let audio_res = args.get_item("audio_data")?;
                        let audio_data = match audio_res {
                            Some(a) => a,
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'audio_data' not found among kwargs passed to constructor, but type was set to 'audio'",
                                ));
                            }
                        };
                        let mime_res = args.get_item("mime_type")?;
                        let mime_type = match mime_res {
                            Some(m) => m,
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'mime_type' not found among kwargs passed to constructor, but type was set to 'audio'",
                                ));
                            }
                        };
                        Ok(MessagePart::AudioPart {
                            r#type: tp,
                            audio_data: audio_data.to_string(),
                            mime_type: mime_type.to_string(),
                        })
                    }
                    PartType::Image => {
                        let image_res = args.get_item("image_data")?;
                        let image_data = match image_res {
                            Some(i) => i.to_string(),
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'image_data' not found among kwargs passed to constructor, but type was set to 'image'",
                                ));
                            }
                        };
                        let mime_res = args.get_item("mime_type")?;
                        let mime_type = match mime_res {
                            Some(m) => {
                                if m.is_none() {
                                    None
                                } else {
                                    Some(m.to_string())
                                }
                            }
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'mime_type' not found among kwargs passed to constructor, but type was set to 'image'",
                                ));
                            }
                        };
                        let is_base64_res = args.get_item("is_base64")?;
                        let is_base64 = match is_base64_res {
                            Some(b) => b.is_truthy()?,
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'is_base64' not found among kwargs passed to constructor, but type was set to 'image'",
                                ));
                            }
                        };
                        Ok(MessagePart::ImagePart {
                            r#type: tp,
                            image_data,
                            is_base64,
                            mime_type,
                        })
                    }
                    PartType::Document => {
                        let doc_res = args.get_item("document_data")?;
                        let document_data = match doc_res {
                            Some(i) => i.to_string(),
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'document_data' not found among kwargs passed to constructor, but type was set to 'document'",
                                ));
                            }
                        };
                        let mime_res = args.get_item("mime_type")?;
                        let mime_type = match mime_res {
                            Some(m) => {
                                if m.is_none() {
                                    None
                                } else {
                                    Some(m.to_string())
                                }
                            }
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'mime_type' not found among kwargs passed to constructor, but type was set to 'document'",
                                ));
                            }
                        };
                        let is_base64_res = args.get_item("is_base64")?;
                        let is_base64 = match is_base64_res {
                            Some(b) => b.is_truthy()?,
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'is_base64' not found among kwargs passed to constructor, but type was set to 'document'",
                                ));
                            }
                        };
                        Ok(MessagePart::DocumentPart {
                            r#type: tp,
                            document_data,
                            is_base64,
                            mime_type,
                        })
                    }
                    PartType::ToolCall => {
                        let name_res = args.get_item("name")?;
                        let name = match name_res {
                            Some(n) => n,
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'name' not found among kwargs passed to constructor, but type was set to 'tool_call'",
                                ));
                            }
                        };
                        let tc_id_res = args.get_item("id")?;
                        let tc_id = match tc_id_res {
                            Some(t) => t,
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'id' not found among kwargs passed to constructor, but type was set to 'tool_call'",
                                ));
                            }
                        };
                        let tc_args_res = args.get_item("arguments")?;
                        let tc_args = match tc_args_res {
                            Some(t) => t,
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'arguments' not found among kwargs passed to constructor, but type was set to 'tool_call'",
                                ));
                            }
                        };
                        Ok(MessagePart::ToolCallPart {
                            r#type: tp,
                            id: tc_id.to_string(),
                            name: name.to_string(),
                            arguments: tc_args.to_string(),
                        })
                    }
                    PartType::ToolResult => {
                        let tc_id_res = args.get_item("tool_call_id")?;
                        let tc_id = match tc_id_res {
                            Some(t) => t,
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'tool_call_id' not found among kwargs passed to constructor, but type was set to 'tool_result'",
                                ));
                            }
                        };
                        let tc_res = args.get_item("result")?;
                        let res = match tc_res {
                            Some(t) => t,
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'result' not found among kwargs passed to constructor, but type was set to 'tool_result'",
                                ));
                            }
                        };
                        Ok(MessagePart::ToolResultPart {
                            r#type: tp,
                            tool_call_id: tc_id.to_string(),
                            result: res.to_string(),
                        })
                    }
                    PartType::Thinking => {
                        let think_res = args.get_item("thinking")?;
                        let thinking = match think_res {
                            Some(t) => t,
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'thinking' not found among kwargs passed to constructor, but type was set to 'thinking'",
                                ));
                            }
                        };
                        let sign_res = args.get_item("signature")?;
                        let signature = match sign_res {
                            Some(t) => {
                                if t.is_none() {
                                    None
                                } else {
                                    Some(t.to_string())
                                }
                            }
                            None => {
                                return Err(PyKeyError::new_err(
                                    "Key 'signature' not found among kwargs passed to constructor, but type was set to 'thinking'",
                                ));
                            }
                        };
                        Ok(MessagePart::ThinkingPart {
                            r#type: tp,
                            thinking: thinking.to_string(),
                            signature,
                        })
                    }
                }
            } else {
                Err(PyValueError::new_err(
                    "Constructor method should have at least one argument passed to it",
                ))
            }
        }

        #[getter]
        #[pyo3(name = "type")]
        fn part_type(&self) -> PartType {
            match self {
                Self::TextPart {
                    r#type: tp,
                    text: _,
                } => tp.to_owned(),
                Self::ThinkingPart {
                    r#type: tp,
                    thinking: _,
                    signature: _,
                } => tp.to_owned(),
                Self::ToolCallPart {
                    r#type: tp,
                    id: _,
                    name: _,
                    arguments: _,
                } => tp.to_owned(),
                Self::ToolResultPart {
                    r#type: tp,
                    tool_call_id: _,
                    result: _,
                } => tp.to_owned(),
                Self::ImagePart {
                    r#type: tp,
                    image_data: _,
                    is_base64: _,
                    mime_type: _,
                } => tp.to_owned(),
                Self::DocumentPart {
                    r#type: tp,
                    document_data: _,
                    is_base64: _,
                    mime_type: _,
                } => tp.to_owned(),
                Self::AudioPart {
                    r#type: tp,
                    audio_data: _,
                    mime_type: _,
                } => tp.to_owned(),
            }
        }

        fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            match self {
                Self::TextPart { text, r#type: tp } => {
                    d.set_item("type", tp.__str__())?;
                    d.set_item("text", text)?;
                }
                Self::ThinkingPart {
                    thinking,
                    signature,
                    r#type: tp,
                } => {
                    d.set_item("type", tp.__str__())?;
                    d.set_item("thinking", thinking)?;
                    d.set_item("signature", signature)?;
                }
                Self::ToolCallPart {
                    id,
                    name,
                    arguments,
                    r#type: tp,
                } => {
                    d.set_item("type", tp.__str__())?;
                    d.set_item("id", id)?;
                    d.set_item("name", name)?;
                    d.set_item("arguments", arguments)?;
                }
                Self::ToolResultPart {
                    tool_call_id,
                    result,
                    r#type: tp,
                } => {
                    d.set_item("type", tp.__str__())?;
                    d.set_item("tool_call_id", tool_call_id)?;
                    d.set_item("result", result)?;
                }
                Self::ImagePart {
                    image_data,
                    is_base64,
                    mime_type,
                    r#type: tp,
                } => {
                    d.set_item("type", tp.__str__())?;
                    d.set_item("image_data", image_data)?;
                    d.set_item("mime_type", mime_type)?;
                    d.set_item("is_base64", is_base64)?;
                }
                Self::DocumentPart {
                    document_data,
                    is_base64,
                    mime_type,
                    r#type: tp,
                } => {
                    d.set_item("type", tp.__str__())?;
                    d.set_item("document_data", document_data)?;
                    d.set_item("mime_type", mime_type)?;
                    d.set_item("is_base64", is_base64)?;
                }
                Self::AudioPart {
                    audio_data,
                    mime_type,
                    r#type: tp,
                } => {
                    d.set_item("type", tp.__str__())?;
                    d.set_item("audio_data", audio_data)?;
                    d.set_item("mime_type", mime_type)?;
                }
            }

            Ok(d)
        }

        #[getter]
        fn text(&self) -> PyResult<String> {
            match self {
                Self::TextPart { r#type: _, text } => Ok(text.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'text' defined for this instance of MessagePart",
                )),
            }
        }

        #[getter]
        fn thinking(&self) -> PyResult<String> {
            match self {
                Self::ThinkingPart {
                    r#type: _,
                    thinking,
                    signature: _,
                } => Ok(thinking.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'thinking' defined for this instance of MessagePart",
                )),
            }
        }

        #[getter]
        fn signature(&self) -> PyResult<Option<String>> {
            match self {
                Self::ThinkingPart {
                    r#type: _,
                    thinking: _,
                    signature,
                } => Ok(signature.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'thinking' defined for this instance of MessagePart",
                )),
            }
        }

        #[getter]
        fn is_base64(&self) -> PyResult<bool> {
            match self {
                Self::ImagePart {
                    r#type: _,
                    is_base64,
                    image_data: _,
                    mime_type: _,
                } => Ok(*is_base64),
                Self::DocumentPart {
                    r#type: _,
                    is_base64,
                    document_data: _,
                    mime_type: _,
                } => Ok(*is_base64),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'is_base64' defined for this instance of MessagePart",
                )),
            }
        }

        #[getter]
        fn document_data(&self) -> PyResult<String> {
            match self {
                Self::DocumentPart {
                    r#type: _,
                    is_base64: _,
                    document_data,
                    mime_type: _,
                } => Ok(document_data.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'document_data' defined for this instance of MessagePart",
                )),
            }
        }

        #[getter]
        fn image_data(&self) -> PyResult<String> {
            match self {
                Self::ImagePart {
                    r#type: _,
                    is_base64: _,
                    image_data,
                    mime_type: _,
                } => Ok(image_data.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'image_data' defined for this instance of MessagePart",
                )),
            }
        }

        #[getter]
        fn audio_data(&self) -> PyResult<String> {
            match self {
                Self::AudioPart {
                    r#type: _,
                    audio_data,
                    mime_type: _,
                } => Ok(audio_data.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'audio_data' defined for this instance of MessagePart",
                )),
            }
        }

        #[getter]
        fn mime_type(&self) -> PyResult<Option<String>> {
            match self {
                Self::AudioPart {
                    r#type: _,
                    audio_data: _,
                    mime_type,
                } => Ok(Some(mime_type.to_owned())),
                Self::ImagePart {
                    r#type: _,
                    image_data: _,
                    is_base64: _,
                    mime_type,
                } => Ok(mime_type.to_owned()),
                Self::DocumentPart {
                    r#type: _,
                    document_data: _,
                    is_base64: _,
                    mime_type,
                } => Ok(mime_type.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'mime_type' defined for this instance of MessagePart",
                )),
            }
        }

        #[getter]
        fn tool_call_id(&self) -> PyResult<String> {
            match self {
                Self::ToolCallPart {
                    r#type: _,
                    id,
                    name: _,
                    arguments: _,
                } => Ok(id.to_owned()),
                Self::ToolResultPart {
                    r#type: _,
                    tool_call_id,
                    result: _,
                } => Ok(tool_call_id.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'tool_call_id' defined for this instance of MessagePart",
                )),
            }
        }

        #[getter]
        fn tool_call_name(&self) -> PyResult<String> {
            match self {
                Self::ToolCallPart {
                    r#type: _,
                    id: _,
                    name,
                    arguments: _,
                } => Ok(name.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'tool_call_name' defined for this instance of MessagePart",
                )),
            }
        }

        #[getter]
        fn tool_call_arguments(&self) -> PyResult<String> {
            match self {
                Self::ToolCallPart {
                    r#type: _,
                    id: _,
                    name: _,
                    arguments,
                } => Ok(arguments.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'tool_call_arguments' defined for this instance of MessagePart",
                )),
            }
        }

        #[getter]
        fn tool_call_result(&self) -> PyResult<String> {
            match self {
                Self::ToolResultPart {
                    r#type: _,
                    tool_call_id: _,
                    result,
                } => Ok(result.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'tool_call_result' defined for this instance of MessagePart",
                )),
            }
        }
    }

    #[pyfunction]
    #[pyo3(signature = (input, /), name = "ImagePart")]
    pub fn image_part(input: Either<String, Vec<u8>>) -> PyResult<MessagePart> {
        match input {
            Either::Left(s) => match Url::parse(&s) {
                Ok(u) => {
                    if matches!(u.scheme(), "http" | "https") {
                        Ok(MessagePart::ImagePart {
                            r#type: PartType::Image,
                            image_data: s,
                            mime_type: None,
                            is_base64: false,
                        })
                    } else {
                        let data = fs::read(&s)?;
                        let format = file_format::FileFormat::from_bytes(&data);
                        if !ALLOWED_IMAGE_TYPES.contains(&format.media_type()) {
                            return Err(PyValueError::new_err(format!(
                                "Unsupported image type: {}. The supported image types are: {}",
                                format.media_type(),
                                ALLOWED_IMAGE_TYPES.join(", ")
                            )));
                        }
                        Ok(MessagePart::ImagePart {
                            r#type: PartType::Image,
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
                        return Err(PyValueError::new_err(format!(
                            "Unsupported image type: {}. The supported image types are: {}",
                            format.media_type(),
                            ALLOWED_IMAGE_TYPES.join(", ")
                        )));
                    }
                    Ok(MessagePart::ImagePart {
                        r#type: PartType::Image,
                        image_data: BASE64_STANDARD.encode(&data),
                        mime_type: Some(format.media_type().to_owned()),
                        is_base64: true,
                    })
                }
            },
            Either::Right(data) => {
                let format = file_format::FileFormat::from_bytes(&data);
                if !ALLOWED_IMAGE_TYPES.contains(&format.media_type()) {
                    return Err(PyValueError::new_err(format!(
                        "Unsupported image type: {}. The supported image types are: {}",
                        format.media_type(),
                        ALLOWED_IMAGE_TYPES.join(", ")
                    )));
                }
                Ok(MessagePart::ImagePart {
                    r#type: PartType::Image,
                    image_data: BASE64_STANDARD.encode(&data),
                    mime_type: Some(format.media_type().to_owned()),
                    is_base64: true,
                })
            }
        }
    }

    #[pyfunction]
    #[pyo3(signature = (input, /), name = "AudioPart")]
    pub fn audio_part(input: Either<String, Vec<u8>>) -> PyResult<MessagePart> {
        match input {
            Either::Left(file) => {
                let intermediate = AudioPart::try_from_file(file)?;
                Ok(MessagePart::AudioPart {
                    r#type: PartType::Audio,
                    audio_data: intermediate.data,
                    mime_type: intermediate.mime_type,
                })
            }
            Either::Right(data) => {
                let intermediate = AudioPart::try_from_bytes(data)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(MessagePart::AudioPart {
                    r#type: PartType::Audio,
                    audio_data: intermediate.data,
                    mime_type: intermediate.mime_type,
                })
            }
        }
    }

    #[pyfunction]
    #[pyo3(signature = (input, /), name = "DocumentPart")]
    pub fn document_part(input: Either<String, Vec<u8>>) -> PyResult<MessagePart> {
        match input {
            Either::Left(s) => match Url::parse(&s) {
                Ok(u) => {
                    if matches!(u.scheme(), "http" | "https") {
                        Ok(MessagePart::DocumentPart {
                            r#type: PartType::Document,
                            document_data: s,
                            mime_type: None,
                            is_base64: false,
                        })
                    } else {
                        let format = file_format::FileFormat::from_file(&s).map_err(|e| {
                            PyValueError::new_err(format!(
                                "Could not infer format from file: {}",
                                e
                            ))
                        })?;
                        if format.media_type() == "application/pdf" {
                            let data = fs::read(&s)?;
                            Ok(MessagePart::DocumentPart {
                                r#type: PartType::Document,
                                document_data: BASE64_STANDARD.encode(data),
                                mime_type: Some("application/pdf".to_string()),
                                is_base64: true,
                            })
                        } else if format.media_type().starts_with("text/") {
                            let data = fs::read_to_string(&s)?;
                            Ok(MessagePart::DocumentPart {
                                r#type: PartType::Document,
                                document_data: data,
                                mime_type: Some("text/plain".to_string()),
                                is_base64: false,
                            })
                        } else {
                            Err(PyValueError::new_err(format!(
                                "Expected either a PDF or a text file, found media type: {}",
                                format.media_type()
                            )))
                        }
                    }
                }
                Err(_) => {
                    let format = file_format::FileFormat::from_file(&s).map_err(|e| {
                        PyValueError::new_err(format!("Could not infer format from file: {}", e))
                    })?;
                    if format.media_type() == "application/pdf" {
                        let data = fs::read(&s)?;
                        Ok(MessagePart::DocumentPart {
                            r#type: PartType::Document,
                            document_data: BASE64_STANDARD.encode(data),
                            mime_type: Some("application/pdf".to_string()),
                            is_base64: true,
                        })
                    } else if format.media_type().starts_with("text/") {
                        let data = fs::read_to_string(&s)?;
                        Ok(MessagePart::DocumentPart {
                            r#type: PartType::Document,
                            document_data: data,
                            mime_type: Some("text/plain".to_string()),
                            is_base64: false,
                        })
                    } else {
                        Err(PyValueError::new_err(format!(
                            "Expected either a PDF or a text file, found media type: {}",
                            format.media_type()
                        )))
                    }
                }
            },
            Either::Right(data) => {
                let format = file_format::FileFormat::from_bytes(&data);
                if format.media_type() != "application/pdf" {
                    return Err(PyValueError::new_err(format!(
                        "Input file should be a PDF, found media type: {}",
                        format.media_type()
                    )));
                }
                Ok(MessagePart::DocumentPart {
                    r#type: PartType::Document,
                    document_data: BASE64_STANDARD.encode(&data),
                    mime_type: Some(format.media_type().to_owned()),
                    is_base64: true,
                })
            }
        }
    }

    #[pyfunction]
    #[pyo3(name = "TextPart")]
    pub fn text_part(text: String) -> MessagePart {
        MessagePart::TextPart {
            r#type: PartType::Text,
            text,
        }
    }

    #[pyfunction]
    #[pyo3(name = "ToolCallPart")]
    pub fn tool_call_part(
        tool_call_id: String,
        function_name: String,
        arguments: String,
    ) -> MessagePart {
        MessagePart::ToolCallPart {
            r#type: PartType::ToolCall,
            id: tool_call_id,
            name: function_name,
            arguments,
        }
    }

    #[pyfunction]
    #[pyo3(name = "ToolResultPart")]
    pub fn tool_result_part(tool_call_id: String, result: String) -> MessagePart {
        MessagePart::ToolResultPart {
            r#type: PartType::ToolResult,
            tool_call_id,
            result,
        }
    }

    #[pyfunction]
    #[pyo3(signature = (thinking, signature = None), name = "ThinkingPart")]
    pub fn thinking_part(thinking: String, signature: Option<String>) -> MessagePart {
        MessagePart::ThinkingPart {
            r#type: PartType::Thinking,
            thinking,
            signature,
        }
    }

    impl From<MessagePart> for NativeMessagePart {
        fn from(value: MessagePart) -> Self {
            match value {
                MessagePart::TextPart { text, r#type: _ } => {
                    NativeMessagePart::Text(TextPart { text })
                }
                MessagePart::ImagePart {
                    image_data,
                    is_base64,
                    mime_type,
                    r#type: _,
                } => NativeMessagePart::Image(ImagePart {
                    data: image_data,
                    is_base64,
                    mime_type,
                }),
                MessagePart::AudioPart {
                    audio_data,
                    mime_type,
                    r#type: _,
                } => NativeMessagePart::Audio(AudioPart {
                    data: audio_data,
                    mime_type,
                }),
                MessagePart::DocumentPart {
                    document_data,
                    mime_type,
                    is_base64,
                    r#type: _,
                } => NativeMessagePart::Document(DocumentPart {
                    data: document_data,
                    mime_type,
                    is_base64,
                }),
                MessagePart::ToolCallPart {
                    id,
                    name,
                    arguments,
                    r#type: _,
                } => NativeMessagePart::ToolCall(ToolCallPart {
                    id,
                    name,
                    arguments,
                }),
                MessagePart::ToolResultPart {
                    tool_call_id,
                    result,
                    r#type: _,
                } => NativeMessagePart::ToolResult(ToolResultPart {
                    tool_call_id,
                    result,
                }),
                MessagePart::ThinkingPart {
                    thinking,
                    signature,
                    r#type: _,
                } => NativeMessagePart::Thinking(ThinkingPart {
                    thinking,
                    signature,
                }),
            }
        }
    }

    impl From<NativeMessagePart> for MessagePart {
        fn from(value: NativeMessagePart) -> Self {
            match value {
                NativeMessagePart::Text(t) => MessagePart::TextPart {
                    r#type: PartType::Text,
                    text: t.text,
                },
                NativeMessagePart::Audio(a) => MessagePart::AudioPart {
                    r#type: PartType::Audio,
                    audio_data: a.data,
                    mime_type: a.mime_type,
                },
                NativeMessagePart::Image(i) => MessagePart::ImagePart {
                    r#type: PartType::Image,
                    image_data: i.data,
                    is_base64: i.is_base64,
                    mime_type: i.mime_type,
                },
                NativeMessagePart::Thinking(t) => MessagePart::ThinkingPart {
                    r#type: PartType::Thinking,
                    thinking: t.thinking,
                    signature: t.signature,
                },
                NativeMessagePart::ToolCall(tc) => MessagePart::ToolCallPart {
                    r#type: PartType::ToolCall,
                    id: tc.id,
                    name: tc.name,
                    arguments: tc.arguments,
                },
                NativeMessagePart::ToolResult(tr) => MessagePart::ToolResultPart {
                    r#type: PartType::ToolResult,
                    tool_call_id: tr.tool_call_id,
                    result: tr.result,
                },
                _ => unreachable!("Unsupported part type"),
            }
        }
    }

    #[pyclass(from_py_object, frozen, eq, hash)]
    #[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
    pub enum MessageRole {
        User,
        Assistant,
        System,
        Tool,
    }

    #[pymethods]
    impl MessageRole {
        #[new]
        fn new(role: String) -> PyResult<Self> {
            match role.as_str() {
                "user" => Ok(Self::User),
                "assistant" => Ok(Self::Assistant),
                "system" => Ok(Self::System),
                "tool" => Ok(Self::Tool),
                _ => Err(PyValueError::new_err(format!(
                    "Unsupported role type: {}",
                    role
                ))),
            }
        }

        fn __repr__(&self) -> String {
            match self {
                Self::User => "user".to_string(),
                Self::Assistant => "assistant".to_string(),
                Self::Tool => "tool".to_string(),
                Self::System => "system".to_string(),
            }
        }

        fn __str__(&self) -> String {
            match self {
                Self::User => "user".to_string(),
                Self::Assistant => "assistant".to_string(),
                Self::Tool => "tool".to_string(),
                Self::System => "system".to_string(),
            }
        }
    }

    impl From<MessageRole> for NativeMessageRole {
        fn from(value: MessageRole) -> Self {
            match value {
                MessageRole::User => Self::User,
                MessageRole::Assistant => Self::Assistant,
                MessageRole::System => Self::System,
                MessageRole::Tool => Self::Tool,
            }
        }
    }

    impl From<NativeMessageRole> for MessageRole {
        fn from(value: NativeMessageRole) -> Self {
            match value {
                NativeMessageRole::User => Self::User,
                NativeMessageRole::Assistant => Self::Assistant,
                NativeMessageRole::System => Self::System,
                NativeMessageRole::Tool => Self::Tool,
            }
        }
    }

    #[pyclass(frozen)]
    #[derive(Debug, FromPyObject)]
    pub struct Message {
        role: MessageRole,
        content: Vec<MessagePart>,
    }

    impl Message {
        fn as_clone(&self) -> Self {
            Self {
                role: self.role,
                content: self.content.iter().map(|c| c.as_clone()).collect(),
            }
        }
    }

    impl From<NativeMessage> for Message {
        fn from(value: NativeMessage) -> Self {
            let mut content: Vec<MessagePart> = vec![];
            for c in value.content {
                content.push(c.into())
            }
            Self {
                role: value.role.into(),
                content,
            }
        }
    }

    #[pymethods]
    impl Message {
        #[new]
        fn new(role: String, content: Vec<MessagePart>) -> PyResult<Self> {
            let rl = MessageRole::new(role)?;
            Ok(Self { role: rl, content })
        }

        #[getter]
        fn role(&self) -> MessageRole {
            self.role
        }

        #[getter]
        fn content(&self) -> Vec<MessagePart> {
            self.content.iter().map(|c| c.as_clone()).collect()
        }

        fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            let content_ls = PyList::empty(py);
            for c in self.content() {
                let pyc = c.to_dict(py)?;
                content_ls.append(pyc)?;
            }
            d.set_item("role", self.role.__str__())?;
            d.set_item("content", content_ls)?;

            Ok(d)
        }
    }

    impl From<Message> for NativeMessage {
        fn from(value: Message) -> Self {
            let role: NativeMessageRole = value.role.into();
            let mut content: Vec<NativeMessagePart> = vec![];
            for c in value.content {
                content.push(c.into())
            }
            Self { role, content }
        }
    }

    #[pyclass(from_py_object, frozen)]
    #[derive(Clone, Debug)]
    pub struct Tool {
        name: String,
        description: String,
        parameters: Schema,
    }

    #[pymethods]
    impl Tool {
        #[new]
        fn new(
            name: String,
            description: String,
            parameters_dict: Bound<'_, PyAny>,
        ) -> PyResult<Self> {
            let value: Value = depythonize(&parameters_dict).map_err(|e| {
                PyValueError::new_err(format!(
                    "`parameters_dict` does not appear to be a JSON object: {}",
                    e
                ))
            })?;
            let schema: Schema = Schema::try_from(value).map_err(|e| {
                PyValueError::new_err(format!(
                    "The provided schema does not seem to be a valid JSON schema: {}",
                    e
                ))
            })?;
            Ok(Self {
                name,
                description,
                parameters: schema,
            })
        }

        #[getter]
        fn name(&self) -> String {
            self.name.clone()
        }

        #[getter]
        fn description(&self) -> String {
            self.description.clone()
        }

        #[getter]
        fn parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let value: Value = self.parameters.clone().into();
            Ok(pythonize(py, &value)?)
        }
    }

    impl From<Tool> for NativeTool {
        fn from(value: Tool) -> Self {
            Self {
                name: value.name,
                description: value.description,
                parameters: value.parameters,
            }
        }
    }

    #[pyclass(from_py_object, frozen)]
    #[derive(Debug, Clone)]
    pub struct OutputFormat {
        name: String,
        description: String,
        schema: Schema,
    }

    #[pymethods]
    impl OutputFormat {
        #[new]
        fn new(name: String, description: String, schema_dict: Bound<'_, PyAny>) -> PyResult<Self> {
            let value: Value = depythonize(&schema_dict).map_err(|e| {
                PyValueError::new_err(format!(
                    "`schema_dict` does not appear to be a JSON object: {}",
                    e
                ))
            })?;
            let schema: Schema = Schema::try_from(value).map_err(|e| {
                PyValueError::new_err(format!(
                    "The provided schema does not seem to be a valid JSON schema: {}",
                    e
                ))
            })?;
            Ok(Self {
                name,
                description,
                schema,
            })
        }

        #[getter]
        fn name(&self) -> String {
            self.name.clone()
        }

        #[getter]
        fn description(&self) -> String {
            self.description.clone()
        }

        #[getter]
        fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let value: Value = self.schema.clone().into();
            Ok(pythonize(py, &value)?)
        }
    }

    impl From<OutputFormat> for NativeOutputFormat {
        fn from(value: OutputFormat) -> Self {
            Self {
                name: value.name,
                description: value.description,
                schema: value.schema,
            }
        }
    }

    #[pyclass(from_py_object, frozen, eq, hash)]
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub enum ApiType {
        OpenAI,
        Anthropic,
    }

    #[pymethods]
    impl ApiType {
        #[new]
        fn new(api_type: String) -> PyResult<Self> {
            match api_type.as_str() {
                "openai" => Ok(Self::OpenAI),
                "anthropic" => Ok(Self::Anthropic),
                _ => Err(PyValueError::new_err(format!(
                    "Unsupported API type: {}",
                    api_type
                ))),
            }
        }

        fn __repr__(&self) -> String {
            match self {
                Self::OpenAI => "openai".to_string(),
                Self::Anthropic => "anthropic".to_string(),
            }
        }

        fn __str__(&self) -> String {
            match self {
                Self::OpenAI => "openai".to_string(),
                Self::Anthropic => "anthropic".to_string(),
            }
        }
    }

    impl From<ApiType> for NativeApiType {
        fn from(value: ApiType) -> Self {
            match value {
                ApiType::Anthropic => Self::Anthropic,
                ApiType::OpenAI => Self::OpenAI,
            }
        }
    }

    #[pyclass(from_py_object, frozen, eq, hash)]
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub enum ReasoningEffort {
        None,
        Minimal,
        Low,
        Medium,
        High,
        Xhigh,
        Maximum,
    }

    #[pymethods]
    impl ReasoningEffort {
        #[new]
        fn new(effort: String) -> PyResult<Self> {
            match effort.as_str() {
                "none" => Ok(Self::None),
                "minimal" => Ok(Self::Minimal),
                "low" => Ok(Self::Low),
                "medium" => Ok(Self::Medium),
                "high" => Ok(Self::High),
                "xhigh" => Ok(Self::Xhigh),
                "max" | "maximum" => Ok(Self::Maximum),
                _ => Err(PyValueError::new_err(format!(
                    "Unsupported reasoning effort type: {}",
                    effort
                ))),
            }
        }

        fn __repr__(&self) -> String {
            match self {
                Self::None => "none".to_string(),
                Self::Minimal => "minimal".to_string(),
                Self::Low => "low".to_string(),
                Self::Medium => "medium".to_string(),
                Self::High => "high".to_string(),
                Self::Xhigh => "xhigh".to_string(),
                Self::Maximum => "maximum".to_string(),
            }
        }

        fn __str__(&self) -> String {
            match self {
                Self::None => "none".to_string(),
                Self::Minimal => "minimal".to_string(),
                Self::Low => "low".to_string(),
                Self::Medium => "medium".to_string(),
                Self::High => "high".to_string(),
                Self::Xhigh => "xhigh".to_string(),
                Self::Maximum => "maximum".to_string(),
            }
        }
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

    #[pyclass(from_py_object, frozen, eq, hash)]
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub enum ToolChoice {
        None,
        Auto,
        Required,
    }

    #[pymethods]
    impl ToolChoice {
        #[new]
        fn new(tool_choice: String) -> PyResult<Self> {
            match tool_choice.as_str() {
                "none" => Ok(Self::None),
                "auto" => Ok(Self::Auto),
                "required" => Ok(Self::Required),
                _ => Err(PyValueError::new_err(format!(
                    "Unsupported tool choice type: {}",
                    tool_choice
                ))),
            }
        }

        fn __repr__(&self) -> String {
            match self {
                Self::Auto => "auto".to_string(),
                Self::None => "none".to_string(),
                Self::Required => "required".to_string(),
            }
        }

        fn __str__(&self) -> String {
            match self {
                Self::Auto => "auto".to_string(),
                Self::None => "none".to_string(),
                Self::Required => "required".to_string(),
            }
        }
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

    #[pyclass]
    #[derive(Debug, FromPyObject)]
    pub struct LLMRequest {
        /// Target API provider.
        #[pyo3(get)]
        pub api_type: ApiType,
        /// Custom base URL for the API. When `None`, the provider's default is used.
        #[pyo3(get)]
        pub base_url: Option<String>,
        /// API key used to authenticate the request.
        #[pyo3(get)]
        pub api_key: String,
        /// Model identifier to use for the request.
        #[pyo3(get)]
        pub model: String,
        /// Conversation history sent to the model.
        pub messages: Vec<Message>,
        /// Maximum number of tokens the model is allowed to generate.
        #[pyo3(get)]
        pub max_output_tokens: Option<u32>,
        /// Sampling temperature (0 = deterministic, higher = more random).
        #[pyo3(get)]
        pub temperature: Option<f32>,
        /// Nucleus sampling parameter (0–1).
        #[pyo3(get)]
        pub top_p: Option<f32>,
        /// Level of reasoning effort requested from the model.
        #[pyo3(get)]
        pub reasoning_effort: Option<ReasoningEffort>,
        /// Prompt cache time-to-live hint (provider-specific format).
        #[pyo3(get)]
        pub prompt_cache_ttl: Option<String>,
        /// Whether to request a streamed response.
        #[pyo3(get)]
        pub stream: bool,
        /// Optional JSON schema for structured outputs.
        #[pyo3(get)]
        pub output_format: Option<OutputFormat>,
        /// Tool definitions made available to the model.
        #[pyo3(get)]
        pub tools: Option<Vec<Tool>>,
        /// Controls whether the model may call tools.
        #[pyo3(get)]
        pub tool_choice: Option<ToolChoice>,
        /// Whether the model may call multiple tools in parallel.
        #[pyo3(get)]
        pub parallel_tool_calls: bool,
    }

    #[pymethods]
    impl LLMRequest {
        #[new]
        #[pyo3(signature = (
            model,
            api_key,
            messages,
            stream,
            api_type = "openai".to_string(),
            base_url = None,
            max_output_tokens = None,
            temperature = None,
            top_p = None,
            reasoning_effort = None,
            prompt_cache_ttl = None,
            output_format = None,
            tool_choice = None,
            tools = None,
            parallel_tool_calls = false,
        ))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            model: String,
            api_key: String,
            messages: Vec<Message>,
            stream: bool,
            api_type: String,
            base_url: Option<String>,
            max_output_tokens: Option<u32>,
            temperature: Option<f32>,
            top_p: Option<f32>,
            reasoning_effort: Option<String>,
            prompt_cache_ttl: Option<String>,
            output_format: Option<OutputFormat>,
            tool_choice: Option<String>,
            tools: Option<Vec<Tool>>,
            parallel_tool_calls: bool,
        ) -> PyResult<Self> {
            let at = ApiType::new(api_type)?;
            let re = match reasoning_effort {
                Some(r) => Some(ReasoningEffort::new(r)?),
                None => None,
            };
            let tc = match tool_choice {
                Some(t) => Some(ToolChoice::new(t)?),
                None => None,
            };
            Ok(Self {
                api_key,
                api_type: at,
                base_url,
                parallel_tool_calls,
                prompt_cache_ttl,
                max_output_tokens,
                messages,
                model,
                tool_choice: tc,
                reasoning_effort: re,
                stream,
                tools,
                output_format,
                temperature,
                top_p,
            })
        }

        #[classmethod]
        #[pyo3(signature = (api_key, messages, model, stream = false))]
        fn from_defaults(
            _cls: &Bound<'_, PyType>,
            api_key: String,
            messages: Vec<Message>,
            model: String,
            stream: bool,
        ) -> Self {
            Self {
                messages,
                model,
                stream,
                api_key,
                api_type: ApiType::OpenAI,
                temperature: None,
                top_p: None,
                max_output_tokens: None,
                output_format: None,
                prompt_cache_ttl: None,
                base_url: Some(DEFAULT_OPENAI_BASE_URL.to_string()),
                reasoning_effort: None,
                tools: None,
                tool_choice: None,
                parallel_tool_calls: false,
            }
        }

        #[getter]
        fn messages(&self) -> Vec<Message> {
            self.messages.iter().map(|m| m.as_clone()).collect()
        }
    }

    impl From<LLMRequest> for NativeLLMRequest {
        fn from(value: LLMRequest) -> Self {
            let mut tools: Option<Vec<NativeTool>> = None;
            if let Some(ts) = value.tools {
                for t in ts {
                    tools.get_or_insert_with(Vec::new).push(t.into());
                }
            }
            let mut output_format: Option<NativeOutputFormat> = None;
            if let Some(of) = value.output_format {
                output_format = Some(of.into())
            }
            let mut messages: Vec<NativeMessage> = vec![];
            for m in value.messages {
                messages.push(m.into());
            }
            Self {
                tools,
                output_format,
                messages,
                max_output_tokens: value.max_output_tokens,
                temperature: value.temperature,
                top_p: value.top_p,
                tool_choice: value.tool_choice.map(|tc| tc.into()),
                parallel_tool_calls: value.parallel_tool_calls,
                prompt_cache_ttl: value.prompt_cache_ttl,
                api_key: value.api_key,
                api_type: value.api_type.into(),
                stream: value.stream,
                reasoning_effort: value.reasoning_effort.map(|r| r.into()),
                model: value.model,
                base_url: value.base_url,
            }
        }
    }

    #[pyclass(from_py_object, frozen, eq)]
    #[derive(Debug, Clone, Eq, PartialEq)]
    pub struct LLMUsage {
        /// Tokens consumed by the prompt.
        #[pyo3(get)]
        pub input_tokens: u32,
        /// Tokens generated in the response.
        #[pyo3(get)]
        pub output_tokens: u32,
        /// Tokens read from a provider cache, when applicable.
        #[pyo3(get)]
        pub cache_read_tokens: Option<u32>,
        /// Tokens written to a provider cache, when applicable.
        #[pyo3(get)]
        pub cache_write_tokens: Option<u32>,
        /// Any additional tokens counted by the provider.
        #[pyo3(get)]
        pub other_tokens: Option<HashMap<String, u32>>,
    }

    impl From<NativeLLMUsage> for LLMUsage {
        fn from(value: NativeLLMUsage) -> Self {
            Self {
                input_tokens: value.input_tokens,
                output_tokens: value.output_tokens,
                cache_read_tokens: value.cache_read_tokens,
                cache_write_tokens: value.cache_write_tokens,
                other_tokens: value.other_tokens,
            }
        }
    }

    #[pyclass(frozen)]
    #[derive(Debug, FromPyObject)]
    pub struct LLMResponse {
        /// Provider-generated response identifier.
        #[pyo3(get)]
        pub id: String,
        /// Unix timestamp of the response, when provided by the API.
        #[pyo3(get)]
        pub created_at: Option<u64>,
        /// The generated message.
        pub message: Message,
        /// Token usage reported for the request.
        #[pyo3(get)]
        pub usage: LLMUsage,
    }

    #[pymethods]
    impl LLMResponse {
        #[new]
        #[pyo3(signature = (id, message, usage, created_at = None))]
        fn new(id: String, message: Message, usage: LLMUsage, created_at: Option<u64>) -> Self {
            Self {
                id,
                message,
                usage,
                created_at,
            }
        }

        #[getter]
        fn message(&self) -> Message {
            self.message.as_clone()
        }
    }

    impl From<NativeLLMResponse> for LLMResponse {
        fn from(value: NativeLLMResponse) -> Self {
            Self {
                id: value.id,
                created_at: value.created_at,
                message: value.message.into(),
                usage: value.usage.into(),
            }
        }
    }

    #[pyclass(from_py_object, frozen, eq)]
    #[derive(Debug, Clone, Copy, Eq, PartialEq)]
    pub struct RetryPolicy {
        /// Maximum number of retries before giving up.
        #[pyo3(get)]
        pub max_retries: u32,
        /// Minimum wait time between retries, in milliseconds.
        #[pyo3(get)]
        pub min_retry_interval: u64,
        /// Maximum wait time between retries, in milliseconds.
        #[pyo3(get)]
        pub max_retry_interval: u64,
        /// Exponential backoff base.
        #[pyo3(get)]
        pub base: u32,
    }

    impl Default for RetryPolicy {
        fn default() -> Self {
            Self {
                max_retries: 3,
                min_retry_interval: 500,
                max_retry_interval: 3000,
                base: 2,
            }
        }
    }

    #[pymethods]
    impl RetryPolicy {
        #[new]
        #[pyo3(signature = (max_retries = 3, min_retry_interval = 500, max_retry_interval = 3000, base = 2))]
        fn new(
            max_retries: u32,
            min_retry_interval: u64,
            max_retry_interval: u64,
            base: u32,
        ) -> Self {
            Self {
                max_retries,
                min_retry_interval,
                max_retry_interval,
                base,
            }
        }
    }

    impl From<RetryPolicy> for NativeRetryPolicy {
        fn from(value: RetryPolicy) -> Self {
            Self {
                max_retries: value.max_retries,
                max_retry_interval: Duration::from_millis(value.max_retry_interval),
                min_retry_interval: Duration::from_millis(value.min_retry_interval),
                base: value.base,
                ..Default::default()
            }
        }
    }

    #[pyclass(from_py_object)]
    #[derive(Default, Clone)]
    #[allow(clippy::upper_case_acronyms)]
    pub struct LLM {
        inner: NativeLLM,
    }

    #[pymethods]
    impl LLM {
        #[new]
        #[pyo3(signature = (retry_policy = None))]
        fn new(retry_policy: Option<RetryPolicy>) -> Self {
            Self {
                inner: NativeLLM::new(retry_policy.unwrap_or_default().into()),
            }
        }

        fn respond<'py>(
            &self,
            py: Python<'py>,
            request: LLMRequest,
        ) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner; // requires NativeLLM: Clone, or Arc it
            future_into_py(py, async move {
                let response = inner.respond(request.into()).await.map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Could not produce the response from the LLM: {}",
                        e
                    ))
                })?;
                Ok(LLMResponse::from(response))
            })
        }
    }
}
