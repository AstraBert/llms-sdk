use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

/// Python bindings for the ``llms_sdk`` Rust crate.
///
/// This module exposes a unified interface for interacting with LLM providers
/// (OpenAI, Anthropic, etc.) from Python.  All types are implemented in Rust
/// and exposed via ``pyo3``.
///
/// Quick start::
///
/// ```python
///     import llms_sdk_py as llm
///
///     req = llm.LLMRequest(
///         model="gpt-4o",
///         api_key="sk-...",
///         messages=[llm.Message("user", [llm.TextPart("Hello!")])],
///         stream=False,
///     )
///     client = llm.LLM()
///     response = await client.respond(req)
/// ```
#[pymodule]
mod llms_sdk_py {
    use futures::stream::{BoxStream, StreamExt, TryStreamExt};
    use llms_sdk::LLMStreamingComplete;
    use llms_sdk::LLMStreamingDelta;
    use llms_sdk::LLMStreamingResponse;
    use llms_sdk::LLMThinkingDelta;
    use llms_sdk::LLMToolDelta;
    use pyo3::exceptions::PyStopAsyncIteration;
    use pyo3_async_runtimes::tokio::future_into_py;
    use pyo3_stub_gen::derive::*;
    use std::collections::HashMap;
    use std::fs;
    use std::sync::Arc;
    use tokio::sync::Mutex;

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
    use std::path::PathBuf;
    use std::time::Duration;
    use url::Url;

    fn path_from_file_url(input: String) -> PyResult<PathBuf> {
        match Url::parse(&input) {
            Ok(url) if url.scheme() == "file" => url
                .to_file_path()
                .map_err(|_| PyValueError::new_err("Invalid file URL")),
            _ => Ok(input.into()),
        }
    }

    /// Discriminator for the different kinds of message parts.
    ///
    /// Use this enum to inspect the ``type`` attribute on a :py:class:`MessagePart`.
    #[gen_stub_pyclass_enum]
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

    /// A single piece of content inside a :py:class:`Message`.
    ///
    /// This is a tagged-union (variant enum).  The ``type`` attribute tells you
    /// which variant is active and therefore which other fields are available::
    ///
    /// ```python
    ///     part = TextPart("hello")
    ///     assert part.type == PartType.Text
    ///     assert part.text == "hello"
    /// ```
    ///
    /// Convenience constructors are provided at module level:
    /// :py:func:`TextPart`, :py:func:`ImagePart`, :py:func:`AudioPart`,
    /// :py:func:`DocumentPart`, :py:func:`ToolCallPart`,
    /// :py:func:`ToolResultPart`, :py:func:`ThinkingPart`.
    #[gen_stub_pyclass_complex_enum]
    #[pyclass(frozen)]
    #[derive(Debug, FromPyObject)]
    #[allow(clippy::enum_variant_names)]
    pub enum MessagePart {
        TextPart {
            #[pyo3(attribute("type"), default = PartType::Text)]
            part_type: PartType,
            text: String,
        },
        ImagePart {
            #[pyo3(attribute("type"), default = PartType::Image)]
            part_type: PartType,
            image_data: String,
            is_base64: bool,
            mime_type: Option<String>,
        },
        AudioPart {
            #[pyo3(attribute("type"), default = PartType::Audio)]
            part_type: PartType,
            audio_data: String,
            mime_type: String,
        },
        DocumentPart {
            #[pyo3(attribute("type"), default = PartType::Document)]
            part_type: PartType,
            document_data: String,
            is_base64: bool,
            mime_type: Option<String>,
        },
        ThinkingPart {
            #[pyo3(attribute("type"), default = PartType::Thinking)]
            part_type: PartType,
            thinking: String,
            signature: Option<String>,
        },
        ToolCallPart {
            #[pyo3(attribute("type"), default = PartType::ToolCall)]
            part_type: PartType,
            id: String,
            name: String,
            arguments: String,
        },
        ToolResultPart {
            #[pyo3(attribute("type"), default = PartType::ToolResult)]
            part_type: PartType,
            tool_call_id: String,
            result: String,
        },
    }

    impl MessagePart {
        fn as_clone(&self) -> Self {
            match self {
                Self::TextPart {
                    text,
                    part_type: tp,
                } => Self::TextPart {
                    part_type: tp.to_owned(),
                    text: text.clone(),
                },
                Self::AudioPart {
                    audio_data,
                    mime_type,
                    part_type: tp,
                } => Self::AudioPart {
                    part_type: tp.to_owned(),
                    audio_data: audio_data.clone(),
                    mime_type: mime_type.clone(),
                },
                Self::ImagePart {
                    image_data,
                    mime_type,
                    part_type: tp,
                    is_base64,
                } => Self::ImagePart {
                    part_type: tp.to_owned(),
                    image_data: image_data.to_owned(),
                    is_base64: *is_base64,
                    mime_type: mime_type.to_owned(),
                },
                Self::DocumentPart {
                    document_data,
                    mime_type,
                    part_type: tp,
                    is_base64,
                } => Self::DocumentPart {
                    part_type: tp.to_owned(),
                    document_data: document_data.to_owned(),
                    is_base64: *is_base64,
                    mime_type: mime_type.to_owned(),
                },
                Self::ToolCallPart {
                    id,
                    name,
                    arguments,
                    part_type: tp,
                } => Self::ToolCallPart {
                    part_type: tp.to_owned(),
                    id: id.clone(),
                    name: name.to_owned(),
                    arguments: arguments.to_owned(),
                },
                Self::ToolResultPart {
                    tool_call_id,
                    result,
                    part_type: tp,
                } => Self::ToolResultPart {
                    part_type: tp.to_owned(),
                    tool_call_id: tool_call_id.clone(),
                    result: result.clone(),
                },
                Self::ThinkingPart {
                    thinking,
                    signature,
                    part_type: tp,
                } => Self::ThinkingPart {
                    part_type: tp.to_owned(),
                    thinking: thinking.clone(),
                    signature: signature.clone(),
                },
            }
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl MessagePart {
        /// Create a :py:class:`MessagePart` from a type string and keyword args.
        ///
        /// Args:
        ///     part_type: One of ``"text"``, ``"image"``, ``"audio"``,
        ///         ``"document"``, ``"thinking"``, ``"tool_call"``,
        ///         ``"tool_result"``.
        ///     **kwargs: Variant-specific fields (see the class docs).
        ///
        /// Returns:
        ///     A new :py:class:`MessagePart` instance.
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
                                part_type: tp,
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
                            part_type: tp,
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
                            part_type: tp,
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
                            part_type: tp,
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
                            part_type: tp,
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
                            part_type: tp,
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
                            part_type: tp,
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

        /// The discriminator telling you which variant this part is.
        #[getter]
        #[pyo3(name = "type")]
        fn part_type(&self) -> PartType {
            match self {
                Self::TextPart {
                    part_type: tp,
                    text: _,
                } => tp.to_owned(),
                Self::ThinkingPart {
                    part_type: tp,
                    thinking: _,
                    signature: _,
                } => tp.to_owned(),
                Self::ToolCallPart {
                    part_type: tp,
                    id: _,
                    name: _,
                    arguments: _,
                } => tp.to_owned(),
                Self::ToolResultPart {
                    part_type: tp,
                    tool_call_id: _,
                    result: _,
                } => tp.to_owned(),
                Self::ImagePart {
                    part_type: tp,
                    image_data: _,
                    is_base64: _,
                    mime_type: _,
                } => tp.to_owned(),
                Self::DocumentPart {
                    part_type: tp,
                    document_data: _,
                    is_base64: _,
                    mime_type: _,
                } => tp.to_owned(),
                Self::AudioPart {
                    part_type: tp,
                    audio_data: _,
                    mime_type: _,
                } => tp.to_owned(),
            }
        }

        /// Convert the part to a plain Python ``dict``.
        fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            match self {
                Self::TextPart {
                    text,
                    part_type: tp,
                } => {
                    d.set_item("type", tp.__str__())?;
                    d.set_item("text", text)?;
                }
                Self::ThinkingPart {
                    thinking,
                    signature,
                    part_type: tp,
                } => {
                    d.set_item("type", tp.__str__())?;
                    d.set_item("thinking", thinking)?;
                    d.set_item("signature", signature)?;
                }
                Self::ToolCallPart {
                    id,
                    name,
                    arguments,
                    part_type: tp,
                } => {
                    d.set_item("type", tp.__str__())?;
                    d.set_item("id", id)?;
                    d.set_item("name", name)?;
                    d.set_item("arguments", arguments)?;
                }
                Self::ToolResultPart {
                    tool_call_id,
                    result,
                    part_type: tp,
                } => {
                    d.set_item("type", tp.__str__())?;
                    d.set_item("tool_call_id", tool_call_id)?;
                    d.set_item("result", result)?;
                }
                Self::ImagePart {
                    image_data,
                    is_base64,
                    mime_type,
                    part_type: tp,
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
                    part_type: tp,
                } => {
                    d.set_item("type", tp.__str__())?;
                    d.set_item("document_data", document_data)?;
                    d.set_item("mime_type", mime_type)?;
                    d.set_item("is_base64", is_base64)?;
                }
                Self::AudioPart {
                    audio_data,
                    mime_type,
                    part_type: tp,
                } => {
                    d.set_item("type", tp.__str__())?;
                    d.set_item("audio_data", audio_data)?;
                    d.set_item("mime_type", mime_type)?;
                }
            }

            Ok(d)
        }

        /// Text payload (only available on ``TextPart``).
        #[getter]
        fn text(&self) -> PyResult<String> {
            match self {
                Self::TextPart { part_type: _, text } => Ok(text.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'text' defined for this instance of MessagePart",
                )),
            }
        }

        /// Reasoning text (only available on ``ThinkingPart``).
        #[getter]
        fn thinking(&self) -> PyResult<String> {
            match self {
                Self::ThinkingPart {
                    part_type: _,
                    thinking,
                    signature: _,
                } => Ok(thinking.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'thinking' defined for this instance of MessagePart",
                )),
            }
        }

        /// Signature block for reasoning (only available on ``ThinkingPart``).
        #[getter]
        fn signature(&self) -> PyResult<Option<String>> {
            match self {
                Self::ThinkingPart {
                    part_type: _,
                    thinking: _,
                    signature,
                } => Ok(signature.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'thinking' defined for this instance of MessagePart",
                )),
            }
        }

        /// Whether the payload is base64-encoded (``ImagePart`` / ``DocumentPart``).
        #[getter]
        fn is_base64(&self) -> PyResult<bool> {
            match self {
                Self::ImagePart {
                    part_type: _,
                    is_base64,
                    image_data: _,
                    mime_type: _,
                } => Ok(*is_base64),
                Self::DocumentPart {
                    part_type: _,
                    is_base64,
                    document_data: _,
                    mime_type: _,
                } => Ok(*is_base64),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'is_base64' defined for this instance of MessagePart",
                )),
            }
        }

        /// Raw document payload or URL (only available on ``DocumentPart``).
        #[getter]
        fn document_data(&self) -> PyResult<String> {
            match self {
                Self::DocumentPart {
                    part_type: _,
                    is_base64: _,
                    document_data,
                    mime_type: _,
                } => Ok(document_data.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'document_data' defined for this instance of MessagePart",
                )),
            }
        }

        /// Raw image payload or URL (only available on ``ImagePart``).
        #[getter]
        fn image_data(&self) -> PyResult<String> {
            match self {
                Self::ImagePart {
                    part_type: _,
                    is_base64: _,
                    image_data,
                    mime_type: _,
                } => Ok(image_data.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'image_data' defined for this instance of MessagePart",
                )),
            }
        }

        /// Raw audio payload (only available on ``AudioPart``).
        #[getter]
        fn audio_data(&self) -> PyResult<String> {
            match self {
                Self::AudioPart {
                    part_type: _,
                    audio_data,
                    mime_type: _,
                } => Ok(audio_data.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'audio_data' defined for this instance of MessagePart",
                )),
            }
        }

        /// MIME type of the payload, when known.
        ///
        /// Available on ``ImagePart``, ``AudioPart`` and ``DocumentPart``.
        #[getter]
        fn mime_type(&self) -> PyResult<Option<String>> {
            match self {
                Self::AudioPart {
                    part_type: _,
                    audio_data: _,
                    mime_type,
                } => Ok(Some(mime_type.to_owned())),
                Self::ImagePart {
                    part_type: _,
                    image_data: _,
                    is_base64: _,
                    mime_type,
                } => Ok(mime_type.to_owned()),
                Self::DocumentPart {
                    part_type: _,
                    document_data: _,
                    is_base64: _,
                    mime_type,
                } => Ok(mime_type.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'mime_type' defined for this instance of MessagePart",
                )),
            }
        }

        /// Tool-call identifier (``ToolCallPart`` / ``ToolResultPart``).
        #[getter]
        fn tool_call_id(&self) -> PyResult<String> {
            match self {
                Self::ToolCallPart {
                    part_type: _,
                    id,
                    name: _,
                    arguments: _,
                } => Ok(id.to_owned()),
                Self::ToolResultPart {
                    part_type: _,
                    tool_call_id,
                    result: _,
                } => Ok(tool_call_id.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'tool_call_id' defined for this instance of MessagePart",
                )),
            }
        }

        /// Name of the tool being called (only ``ToolCallPart``).
        #[getter]
        fn tool_call_name(&self) -> PyResult<String> {
            match self {
                Self::ToolCallPart {
                    part_type: _,
                    id: _,
                    name,
                    arguments: _,
                } => Ok(name.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'tool_call_name' defined for this instance of MessagePart",
                )),
            }
        }

        /// JSON-encoded arguments for the tool call (only ``ToolCallPart``).
        #[getter]
        fn tool_call_arguments(&self) -> PyResult<String> {
            match self {
                Self::ToolCallPart {
                    part_type: _,
                    id: _,
                    name: _,
                    arguments,
                } => Ok(arguments.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'tool_call_arguments' defined for this instance of MessagePart",
                )),
            }
        }

        /// Result returned by a tool execution (only ``ToolResultPart``).
        #[getter]
        fn tool_call_result(&self) -> PyResult<String> {
            match self {
                Self::ToolResultPart {
                    part_type: _,
                    tool_call_id: _,
                    result,
                } => Ok(result.to_owned()),
                _ => Err(PyAttributeError::new_err(
                    "No attribute 'tool_call_result' defined for this instance of MessagePart",
                )),
            }
        }
    }

    /// Convenience constructor for an image part.
    ///
    /// Args:
    ///     input: Either a file path, file/HTTP URL string, or raw ``bytes``.
    ///
    /// Returns:
    ///     A :py:class:`MessagePart` configured as ``ImagePart``.
    ///
    /// Raises:
    ///     ValueError: If the image format is not supported.
    #[gen_stub_pyfunction]
    #[pyfunction]
    #[pyo3(signature = (input, /), name = "ImagePart")]
    pub fn image_part(py: Python<'_>, input: Either<String, Vec<u8>>) -> PyResult<MessagePart> {
        py.detach(move || match input {
            Either::Left(s) => match Url::parse(&s) {
                Ok(url) if matches!(url.scheme(), "http" | "https") => Ok(MessagePart::ImagePart {
                    part_type: PartType::Image,
                    image_data: s,
                    mime_type: None,
                    is_base64: false,
                }),
                _ => {
                    let data = fs::read(path_from_file_url(s)?)?;
                    let format = file_format::FileFormat::from_bytes(&data);
                    if !ALLOWED_IMAGE_TYPES.contains(&format.media_type()) {
                        return Err(PyValueError::new_err(format!(
                            "Unsupported image type: {}. The supported image types are: {}",
                            format.media_type(),
                            ALLOWED_IMAGE_TYPES.join(", ")
                        )));
                    }
                    Ok(MessagePart::ImagePart {
                        part_type: PartType::Image,
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
                    part_type: PartType::Image,
                    image_data: BASE64_STANDARD.encode(&data),
                    mime_type: Some(format.media_type().to_owned()),
                    is_base64: true,
                })
            }
        })
    }

    /// Convenience constructor for an audio part.
    ///
    /// Args:
    ///     input: Either a file path or file URL string, or raw ``bytes``.
    ///
    /// Returns:
    ///     A :py:class:`MessagePart` configured as ``AudioPart``.
    #[gen_stub_pyfunction]
    #[pyfunction]
    #[pyo3(signature = (input, /), name = "AudioPart")]
    pub fn audio_part(py: Python<'_>, input: Either<String, Vec<u8>>) -> PyResult<MessagePart> {
        py.detach(move || match input {
            Either::Left(file) => {
                let intermediate = AudioPart::try_from_file(path_from_file_url(file)?)?;
                Ok(MessagePart::AudioPart {
                    part_type: PartType::Audio,
                    audio_data: intermediate.data,
                    mime_type: intermediate.mime_type,
                })
            }
            Either::Right(data) => {
                let intermediate = AudioPart::try_from_bytes(data)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(MessagePart::AudioPart {
                    part_type: PartType::Audio,
                    audio_data: intermediate.data,
                    mime_type: intermediate.mime_type,
                })
            }
        })
    }

    /// Convenience constructor for a document part.
    ///
    /// Supports PDF and plain-text files.  When *input* is a path the MIME
    /// type is inferred automatically; raw ``bytes`` must be a PDF.
    ///
    /// Args:
    ///     input: Either a file path, file/HTTP URL string, or raw ``bytes``.
    ///
    /// Returns:
    ///     A :py:class:`MessagePart` configured as ``DocumentPart``.
    #[gen_stub_pyfunction]
    #[pyfunction]
    #[pyo3(signature = (input, /), name = "DocumentPart")]
    pub fn document_part(py: Python<'_>, input: Either<String, Vec<u8>>) -> PyResult<MessagePart> {
        py.detach(move || match input {
            Either::Left(s) => match Url::parse(&s) {
                Ok(url) if matches!(url.scheme(), "http" | "https") => {
                    Ok(MessagePart::DocumentPart {
                        part_type: PartType::Document,
                        document_data: s,
                        mime_type: None,
                        is_base64: false,
                    })
                }
                _ => {
                    let file = path_from_file_url(s)?;
                    let format = file_format::FileFormat::from_file(&file).map_err(|e| {
                        PyValueError::new_err(format!("Could not infer format from file: {}", e))
                    })?;
                    if format.media_type() == "application/pdf" {
                        let data = fs::read(file)?;
                        Ok(MessagePart::DocumentPart {
                            part_type: PartType::Document,
                            document_data: BASE64_STANDARD.encode(data),
                            mime_type: Some("application/pdf".to_string()),
                            is_base64: true,
                        })
                    } else if format.media_type().starts_with("text/") {
                        let data = fs::read_to_string(file)?;
                        Ok(MessagePart::DocumentPart {
                            part_type: PartType::Document,
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
                    part_type: PartType::Document,
                    document_data: BASE64_STANDARD.encode(&data),
                    mime_type: Some(format.media_type().to_owned()),
                    is_base64: true,
                })
            }
        })
    }

    /// Convenience constructor for a text part.
    ///
    /// Args:
    ///     text: The text payload.
    ///
    /// Returns:
    ///     A :py:class:`MessagePart` configured as ``TextPart``.
    #[gen_stub_pyfunction]
    #[pyfunction]
    #[pyo3(name = "TextPart")]
    pub fn text_part(text: String) -> MessagePart {
        MessagePart::TextPart {
            part_type: PartType::Text,
            text,
        }
    }

    /// Convenience constructor for a tool-call part.
    ///
    /// Args:
    ///     tool_call_id: Unique identifier for this tool call.
    ///     function_name: Name of the tool / function to invoke.
    ///     arguments: JSON-encoded argument string.
    ///
    /// Returns:
    ///     A :py:class:`MessagePart` configured as ``ToolCallPart``.
    #[gen_stub_pyfunction]
    #[pyfunction]
    #[pyo3(name = "ToolCallPart")]
    pub fn tool_call_part(
        tool_call_id: String,
        function_name: String,
        arguments: String,
    ) -> MessagePart {
        MessagePart::ToolCallPart {
            part_type: PartType::ToolCall,
            id: tool_call_id,
            name: function_name,
            arguments,
        }
    }

    /// Convenience constructor for a tool-result part.
    ///
    /// Args:
    ///     tool_call_id: Identifier matching the original :py:func:`ToolCallPart`.
    ///     result: String result produced by the tool.
    ///
    /// Returns:
    ///     A :py:class:`MessagePart` configured as ``ToolResultPart``.
    #[gen_stub_pyfunction]
    #[pyfunction]
    #[pyo3(name = "ToolResultPart")]
    pub fn tool_result_part(tool_call_id: String, result: String) -> MessagePart {
        MessagePart::ToolResultPart {
            part_type: PartType::ToolResult,
            tool_call_id,
            result,
        }
    }

    /// Convenience constructor for a reasoning / thinking part.
    ///
    /// Args:
    ///     thinking: The model's internal reasoning text.
    ///     signature: Optional signature block (provider-specific).
    ///
    /// Returns:
    ///     A :py:class:`MessagePart` configured as ``ThinkingPart``.
    #[gen_stub_pyfunction]
    #[pyfunction]
    #[pyo3(signature = (thinking, signature = None), name = "ThinkingPart")]
    pub fn thinking_part(thinking: String, signature: Option<String>) -> MessagePart {
        MessagePart::ThinkingPart {
            part_type: PartType::Thinking,
            thinking,
            signature,
        }
    }

    impl From<MessagePart> for NativeMessagePart {
        fn from(value: MessagePart) -> Self {
            match value {
                MessagePart::TextPart { text, part_type: _ } => {
                    NativeMessagePart::Text(TextPart { text })
                }
                MessagePart::ImagePart {
                    image_data,
                    is_base64,
                    mime_type,
                    part_type: _,
                } => NativeMessagePart::Image(ImagePart {
                    data: image_data,
                    is_base64,
                    mime_type,
                }),
                MessagePart::AudioPart {
                    audio_data,
                    mime_type,
                    part_type: _,
                } => NativeMessagePart::Audio(AudioPart {
                    data: audio_data,
                    mime_type,
                }),
                MessagePart::DocumentPart {
                    document_data,
                    mime_type,
                    is_base64,
                    part_type: _,
                } => NativeMessagePart::Document(DocumentPart {
                    data: document_data,
                    mime_type,
                    is_base64,
                }),
                MessagePart::ToolCallPart {
                    id,
                    name,
                    arguments,
                    part_type: _,
                } => NativeMessagePart::ToolCall(ToolCallPart {
                    id,
                    name,
                    arguments,
                }),
                MessagePart::ToolResultPart {
                    tool_call_id,
                    result,
                    part_type: _,
                } => NativeMessagePart::ToolResult(ToolResultPart {
                    tool_call_id,
                    result,
                }),
                MessagePart::ThinkingPart {
                    thinking,
                    signature,
                    part_type: _,
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
                    part_type: PartType::Text,
                    text: t.text,
                },
                NativeMessagePart::Audio(a) => MessagePart::AudioPart {
                    part_type: PartType::Audio,
                    audio_data: a.data,
                    mime_type: a.mime_type,
                },
                NativeMessagePart::Image(i) => MessagePart::ImagePart {
                    part_type: PartType::Image,
                    image_data: i.data,
                    is_base64: i.is_base64,
                    mime_type: i.mime_type,
                },
                NativeMessagePart::Thinking(t) => MessagePart::ThinkingPart {
                    part_type: PartType::Thinking,
                    thinking: t.thinking,
                    signature: t.signature,
                },
                NativeMessagePart::ToolCall(tc) => MessagePart::ToolCallPart {
                    part_type: PartType::ToolCall,
                    id: tc.id,
                    name: tc.name,
                    arguments: tc.arguments,
                },
                NativeMessagePart::ToolResult(tr) => MessagePart::ToolResultPart {
                    part_type: PartType::ToolResult,
                    tool_call_id: tr.tool_call_id,
                    result: tr.result,
                },
                _ => unreachable!("Unsupported part type"),
            }
        }
    }

    /// Role of a participant in the conversation.
    #[gen_stub_pyclass_enum]
    #[pyclass(from_py_object, frozen, eq, hash)]
    #[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
    pub enum MessageRole {
        User,
        Assistant,
        System,
        Tool,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl MessageRole {
        /// Parse a role from its string representation.
        ///
        /// Args:
        ///     role: One of ``"user"``, ``"assistant"``, ``"system"``, ``"tool"``.
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

    /// A single message in the conversation history.
    ///
    /// Consists of a :py:class:`MessageRole` and a list of :py:class:`MessagePart`s.
    #[gen_stub_pyclass]
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

    #[gen_stub_pymethods]
    #[pymethods]
    impl Message {
        /// Create a new message.
        ///
        /// Args:
        ///     role: Conversation role (see :py:class:`MessageRole`).
        ///     content: List of :py:class:`MessagePart` objects.
        #[new]
        fn new(role: String, content: Vec<MessagePart>) -> PyResult<Self> {
            let rl = MessageRole::new(role)?;
            Ok(Self { role: rl, content })
        }

        /// The role of this message.
        #[getter]
        fn role(&self) -> MessageRole {
            self.role
        }

        /// The content parts that make up this message.
        #[getter]
        fn content(&self) -> Vec<MessagePart> {
            self.content.iter().map(|c| c.as_clone()).collect()
        }

        /// Convert the message to a plain Python ``dict``.
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

    /// Definition of a tool (function) that the model may call.
    ///
    /// The *parameters_dict* must be a valid JSON Schema describing the
    /// arguments the tool accepts.
    #[gen_stub_pyclass]
    #[pyclass(from_py_object, frozen)]
    #[derive(Clone, Debug)]
    pub struct Tool {
        name: String,
        description: String,
        parameters: Schema,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl Tool {
        /// Create a new tool definition.
        ///
        /// Args:
        ///     name: Tool / function name.
        ///     description: Human-readable description for the model.
        ///     parameters_dict: A Python ``dict`` that is a valid JSON Schema.
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

        /// Tool name.
        #[getter]
        fn name(&self) -> String {
            self.name.clone()
        }

        /// Tool description.
        #[getter]
        fn description(&self) -> String {
            self.description.clone()
        }

        /// JSON Schema for the tool's parameters (as a Python ``dict``).
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

    /// Structured-output format definition.
    ///
    /// When supplied in a :py:class:`LLMRequest`, the model is asked to produce
    /// output conforming to the given JSON Schema.
    #[gen_stub_pyclass]
    #[pyclass(from_py_object, frozen)]
    #[derive(Debug, Clone)]
    pub struct OutputFormat {
        name: String,
        description: String,
        schema: Schema,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl OutputFormat {
        /// Create a new output format.
        ///
        /// Args:
        ///     name: Short name for the schema.
        ///     description: Description of what the schema represents.
        ///     schema_dict: A Python ``dict`` that is a valid JSON Schema.
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

        /// Format name.
        #[getter]
        fn name(&self) -> String {
            self.name.clone()
        }

        /// Format description.
        #[getter]
        fn description(&self) -> String {
            self.description.clone()
        }

        /// JSON Schema as a Python ``dict``.
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

    /// Supported LLM API providers.
    #[gen_stub_pyclass_enum]
    #[pyclass(from_py_object, frozen, eq, hash)]
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub enum ApiType {
        OpenAI,
        Anthropic,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl ApiType {
        /// Parse an API type from its string representation.
        ///
        /// Args:
        ///     api_type: ``"openai"`` or ``"anthropic"``.
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

    /// Controls how much reasoning the model should perform.
    #[gen_stub_pyclass_enum]
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

    #[gen_stub_pymethods]
    #[pymethods]
    impl ReasoningEffort {
        /// Parse a reasoning-effort level from its string representation.
        ///
        /// Args:
        ///     effort: One of ``"none"``, ``"minimal"``, ``"low"``, ``"medium"``,
        ///         ``"high"``, ``"xhigh"``, ``"max"`` / ``"maximum"``.
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

    /// Controls whether the model is allowed to call tools.
    #[gen_stub_pyclass_enum]
    #[pyclass(from_py_object, frozen, eq, hash)]
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub enum ToolChoice {
        None,
        Auto,
        Required,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl ToolChoice {
        /// Parse a tool-choice policy from its string representation.
        ///
        /// Args:
        ///     tool_choice: ``"none"``, ``"auto"`` or ``"required"``.
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

    /// Configuration for a single LLM request.
    ///
    /// All fields have sensible defaults where applicable.  The only required
    /// arguments at construction time are *model*, *api_key*, *messages* and
    /// *stream*.
    #[gen_stub_pyclass]
    #[pyclass]
    #[derive(Debug, FromPyObject)]
    pub struct LLMRequest {
        /// Target API provider.
        #[pyo3(get)]
        pub api_type: ApiType,
        /// Custom base URL for the API. When ``None``, the provider's default is used.
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

    #[gen_stub_pymethods]
    #[pymethods]
    impl LLMRequest {
        /// Build a request with explicit parameters.
        ///
        /// Args:
        ///     model: Model identifier (e.g. ``"gpt-4o"``).
        ///     api_key: Provider API key.
        ///     messages: List of :py:class:`Message` objects.
        ///     stream: Whether to request a streaming response.
        ///     api_type: ``"openai"`` (default) or ``"anthropic"``.
        ///     base_url: Override the provider's default base URL.
        ///     max_output_tokens: Hard limit on generated tokens.
        ///     temperature: Sampling temperature (0 = deterministic).
        ///     top_p: Nucleus-sampling threshold.
        ///     reasoning_effort: Reasoning level as a string (see :py:class:`ReasoningEffort`).
        ///     prompt_cache_ttl: Provider-specific cache TTL hint.
        ///     output_format: Structured-output schema (see :py:class:`OutputFormat`).
        ///     tool_choice: Tool-calling policy as a string (see :py:class:`ToolChoice`).
        ///     tools: List of :py:class:`Tool` definitions.
        ///     parallel_tool_calls: Allow the model to call multiple tools at once.
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

        /// Create a request using OpenAI-compatible defaults.
        ///
        /// Args:
        ///     api_key: Provider API key.
        ///     messages: List of :py:class:`Message` objects.
        ///     model: Model identifier.
        ///     stream: Whether to request a streaming response (default ``False``).
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

        /// Conversation messages.
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

    /// Token-usage statistics for a completed request.
    #[gen_stub_pyclass]
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

    #[gen_stub_pymethods]
    #[pymethods]
    impl LLMUsage {
        /// Create a usage object.
        ///
        /// Args:
        ///     input_tokens: Prompt tokens consumed.
        ///     output_tokens: Tokens generated.
        ///     cache_read_tokens: Tokens served from cache (optional).
        ///     cache_write_tokens: Tokens written to cache (optional).
        ///     other_tokens: Provider-specific extra counters (optional).
        #[new]
        fn new(
            input_tokens: u32,
            output_tokens: u32,
            cache_read_tokens: Option<u32>,
            cache_write_tokens: Option<u32>,
            other_tokens: Option<HashMap<String, u32>>,
        ) -> Self {
            Self {
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_write_tokens,
                other_tokens,
            }
        }

        /// Convert to a plain Python ``dict``.
        fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            d.set_item("input_tokens", self.input_tokens)?;
            d.set_item("output_tokens", self.output_tokens)?;
            d.set_item("cache_read_tokens", self.cache_read_tokens)?;
            d.set_item("cache_write_tokens", self.cache_write_tokens)?;
            d.set_item("other_tokens", &self.other_tokens)?;

            Ok(d)
        }
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

    /// Non-streaming response from an LLM.
    #[gen_stub_pyclass]
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

    #[gen_stub_pymethods]
    #[pymethods]
    impl LLMResponse {
        /// Create a response object.
        ///
        /// Args:
        ///     id: Provider-generated response ID.
        ///     message: The generated :py:class:`Message`.
        ///     usage: :py:class:`LLMUsage` statistics.
        ///     created_at: Unix timestamp (optional).
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

        /// The generated message.
        #[getter]
        fn message(&self) -> Message {
            self.message.as_clone()
        }

        /// Convert to a plain Python ``dict``.
        fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);

            d.set_item("id", self.id.clone())?;
            d.set_item("message", self.message.to_dict(py)?)?;
            d.set_item("usage", self.usage.to_dict(py)?)?;
            d.set_item("created_at", self.created_at)?;

            Ok(d)
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

    /// A text delta emitted while streaming a response.
    #[gen_stub_pyclass]
    #[pyclass(from_py_object, frozen, eq)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct StreamTextPart {
        /// Identifier of the response this delta belongs to.
        #[pyo3(get)]
        response_id: String,
        /// Unix timestamp of the response, when provided by the API.
        #[pyo3(get)]
        created_at: Option<u64>,
        /// Chunk of generated text, if any.
        #[pyo3(get)]
        text_delta: Option<String>,
        /// Whether this delta signals the end of the stream.
        #[pyo3(get)]
        stop: bool,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl StreamTextPart {
        /// Create a text-delta object.
        ///
        /// Args:
        ///     response_id: Parent response identifier.
        ///     created_at: Unix timestamp (optional).
        ///     text_delta: Chunk of text, if any.
        ///     stop: ``True`` when this is the final delta.
        #[new]
        fn new(
            response_id: String,
            created_at: Option<u64>,
            text_delta: Option<String>,
            stop: bool,
        ) -> Self {
            Self {
                response_id,
                created_at,
                text_delta,
                stop,
            }
        }

        /// Convert to a plain Python ``dict``.
        fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            d.set_item("response_id", self.response_id.clone())?;
            d.set_item("created_at", self.created_at)?;
            d.set_item("text_delta", self.text_delta.clone())?;
            d.set_item("stop", self.stop)?;

            Ok(d)
        }
    }

    /// A reasoning / thinking delta emitted while streaming.
    #[gen_stub_pyclass]
    #[pyclass(from_py_object, frozen, eq)]
    #[derive(Debug, Clone, Eq, PartialEq)]
    pub struct StreamThinkingPart {
        /// Identifier of the response this delta belongs to.
        #[pyo3(get)]
        response_id: String,
        /// Unix timestamp of the response, when provided by the API.
        #[pyo3(get)]
        created_at: Option<u64>,
        /// Chunk of reasoning text, if any.
        #[pyo3(get)]
        thinking_delta: Option<String>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl StreamThinkingPart {
        /// Create a thinking-delta object.
        ///
        /// Args:
        ///     response_id: Parent response identifier.
        ///     created_at: Unix timestamp (optional).
        ///     thinking_delta: Chunk of reasoning text, if any.
        #[new]
        fn new(
            response_id: String,
            created_at: Option<u64>,
            thinking_delta: Option<String>,
        ) -> Self {
            Self {
                response_id,
                created_at,
                thinking_delta,
            }
        }

        /// Convert to a plain Python ``dict``.
        fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            d.set_item("response_id", self.response_id.clone())?;
            d.set_item("created_at", self.created_at)?;
            d.set_item("thinking_delta", self.thinking_delta.clone())?;

            Ok(d)
        }
    }

    /// An in-progress tool-call delta emitted while streaming.
    #[gen_stub_pyclass]
    #[pyclass(from_py_object, frozen, eq)]
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct StreamToolCallPart {
        /// Identifier for the completion response the tool call belongs to.
        #[pyo3(get)]
        response_id: String,
        /// Identifier for the in-progress tool call.
        #[pyo3(get)]
        tool_call_id: String,
        /// Name of the tool being called.
        #[pyo3(get)]
        name: String,
        /// Partial JSON arguments accumulated so far.
        #[pyo3(get)]
        partial_arguments: String,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl StreamToolCallPart {
        /// Create a tool-call delta object.
        ///
        /// Args:
        ///     response_id: Parent response identifier.
        ///     tool_call_id: Identifier for this tool call.
        ///     name: Tool name.
        ///     partial_arguments: JSON arguments accumulated so far.
        #[new]
        fn new(
            response_id: String,
            tool_call_id: String,
            name: String,
            partial_arguments: String,
        ) -> Self {
            Self {
                response_id,
                tool_call_id,
                name,
                partial_arguments,
            }
        }

        /// Convert to a plain Python ``dict``.
        fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            d.set_item("response_id", self.response_id.clone())?;
            d.set_item("tool_call_id", self.tool_call_id.clone())?;
            d.set_item("name", self.name.clone())?;
            d.set_item("partial_arguments", self.partial_arguments.clone())?;

            Ok(d)
        }
    }

    /// Final part of a streaming response, carrying the assembled result.
    #[gen_stub_pyclass]
    #[pyclass(frozen)]
    #[derive(Debug, FromPyObject)]
    pub struct StreamEndPart {
        /// Provider-generated response identifier.
        #[pyo3(get)]
        id: String,
        /// Unix timestamp of the response, when provided by the API.
        #[pyo3(get)]
        created_at: Option<u64>,
        /// All text deltas that make up the response.
        #[pyo3(get)]
        deltas: Vec<StreamTextPart>,
        /// All reasoning deltas, if the model produced any.
        #[pyo3(get)]
        thinking_deltas: Option<Vec<StreamThinkingPart>>,
        /// Token usage reported for the request, if provided.
        #[pyo3(get)]
        usage: Option<LLMUsage>,
        /// Full message, with text, thinking and parsed tool calls
        message: Message,
    }

    impl StreamEndPart {
        fn as_clone(&self) -> Self {
            Self {
                id: self.id.clone(),
                created_at: self.created_at,
                deltas: self.deltas.clone(),
                thinking_deltas: self.thinking_deltas.clone(),
                usage: self.usage.clone(),
                message: self.message.as_clone(),
            }
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl StreamEndPart {
        /// Create a stream-end object.
        ///
        /// Args:
        ///     id: Provider-generated response ID.
        ///     created_at: Unix timestamp (optional).
        ///     deltas: All :py:class:`StreamTextPart` deltas.
        ///     thinking_deltas: All :py:class:`StreamThinkingPart` deltas (optional).
        ///     usage: Final :py:class:`LLMUsage` (optional).
        ///     message: Assembled :py:class:`Message`.
        #[new]
        fn new(
            id: String,
            created_at: Option<u64>,
            deltas: Vec<StreamTextPart>,
            thinking_deltas: Option<Vec<StreamThinkingPart>>,
            usage: Option<LLMUsage>,
            message: Message,
        ) -> Self {
            Self {
                id,
                created_at,
                deltas,
                thinking_deltas,
                usage,
                message,
            }
        }

        /// The fully assembled message.
        #[getter]
        fn message(&self) -> Message {
            self.message.as_clone()
        }

        /// Convert to a plain Python ``dict``.
        fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            let deltas = self
                .deltas
                .iter()
                .map(|delta| delta.to_dict(py))
                .collect::<PyResult<Vec<_>>>()?;
            let thinking_deltas = self
                .thinking_deltas
                .as_ref()
                .map(|deltas| {
                    deltas
                        .iter()
                        .map(|delta| delta.to_dict(py))
                        .collect::<PyResult<Vec<_>>>()
                })
                .transpose()?;
            let usage = self
                .usage
                .as_ref()
                .map(|usage| usage.to_dict(py))
                .transpose()?;
            d.set_item("id", self.id.clone())?;
            d.set_item("created_at", self.created_at)?;
            d.set_item("deltas", deltas)?;
            d.set_item("thinking_deltas", thinking_deltas)?;
            d.set_item("message", self.message.to_dict(py)?)?;
            d.set_item("usage", usage)?;

            Ok(d)
        }
    }

    /// Discriminator for the different kinds of streaming parts.
    #[gen_stub_pyclass_enum]
    #[pyclass(from_py_object, frozen, eq, hash)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum StreamPartType {
        Text,
        Thinking,
        ToolCall,
        End,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl StreamPartType {
        /// Parse a stream-part type from its string representation.
        ///
        /// Args:
        ///     part_type: ``"text"``, ``"thinking"``, ``"tool_call"`` or ``"end"``.
        #[new]
        fn new(part_type: String) -> PyResult<Self> {
            match part_type.as_str() {
                "text" => Ok(Self::Text),
                "thinking" => Ok(Self::Thinking),
                "tool_call" => Ok(Self::ToolCall),
                "end" => Ok(Self::End),
                _ => Err(PyValueError::new_err(format!(
                    "Unsupported streaming part type: {}",
                    part_type
                ))),
            }
        }

        fn __str__(&self) -> String {
            match self {
                Self::Text => "text".to_string(),
                Self::Thinking => "thinking".to_string(),
                Self::ToolCall => "tool_call".to_string(),
                Self::End => "end".to_string(),
            }
        }

        fn __repr__(&self) -> String {
            match self {
                Self::Text => "text".to_string(),
                Self::Thinking => "thinking".to_string(),
                Self::ToolCall => "tool_call".to_string(),
                Self::End => "end".to_string(),
            }
        }
    }

    /// A single item yielded by an async streaming response.
    ///
    /// Inspect ``type`` to determine which optional field is populated.
    #[gen_stub_pyclass]
    #[pyclass(frozen)]
    #[derive(Debug, FromPyObject)]
    pub struct StreamPart {
        #[pyo3(get, name = "type")]
        part_type: StreamPartType,
        #[pyo3(get)]
        text: Option<StreamTextPart>,
        #[pyo3(get)]
        thinking: Option<StreamThinkingPart>,
        #[pyo3(get)]
        tool_call: Option<StreamToolCallPart>,
        end: Option<StreamEndPart>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl StreamPart {
        /// Create a stream part.
        ///
        /// Args:
        ///     part_type: The discriminator (:py:class:`StreamPartType`).
        ///     text: Populated when ``type == StreamPartType.Text``.
        ///     tool_call: Populated when ``type == StreamPartType.ToolCall``.
        ///     thinking: Populated when ``type == StreamPartType.Thinking``.
        ///     end: Populated when ``type == StreamPartType.End``.
        #[new]
        fn new(
            part_type: StreamPartType,
            text: Option<StreamTextPart>,
            tool_call: Option<StreamToolCallPart>,
            thinking: Option<StreamThinkingPart>,
            end: Option<StreamEndPart>,
        ) -> Self {
            Self {
                part_type,
                text,
                tool_call,
                thinking,
                end,
            }
        }

        /// The end-of-stream payload (only when ``type == StreamPartType.End``).
        #[getter]
        fn end(&self) -> Option<StreamEndPart> {
            self.end.as_ref().map(|m| m.as_clone())
        }

        /// Convert to a plain Python ``dict``.
        fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
            let d = PyDict::new(py);
            let text = self
                .text
                .as_ref()
                .map(|text| text.to_dict(py))
                .transpose()?;
            let thinking = self
                .thinking
                .as_ref()
                .map(|thinking| thinking.to_dict(py))
                .transpose()?;
            let tool_call = self
                .tool_call
                .as_ref()
                .map(|tool_call| tool_call.to_dict(py))
                .transpose()?;
            let end = self.end.as_ref().map(|end| end.to_dict(py)).transpose()?;
            d.set_item("type", self.part_type.__str__())?;
            d.set_item("text", text)?;
            d.set_item("thinking", thinking)?;
            d.set_item("tool_call", tool_call)?;
            d.set_item("end", end)?;

            Ok(d)
        }
    }

    impl From<LLMStreamingDelta> for StreamPart {
        fn from(value: LLMStreamingDelta) -> Self {
            Self {
                part_type: StreamPartType::Text,
                text: Some(StreamTextPart {
                    text_delta: value.delta,
                    response_id: value.response_id,
                    created_at: value.created_at,
                    stop: value.stop,
                }),
                end: None,
                tool_call: None,
                thinking: None,
            }
        }
    }

    impl From<LLMThinkingDelta> for StreamPart {
        fn from(value: LLMThinkingDelta) -> Self {
            Self {
                part_type: StreamPartType::Thinking,
                text: None,
                tool_call: None,
                end: None,
                thinking: Some(StreamThinkingPart {
                    response_id: value.response_id,
                    created_at: value.created_at,
                    thinking_delta: value.delta,
                }),
            }
        }
    }

    impl From<LLMToolDelta> for StreamPart {
        fn from(value: LLMToolDelta) -> Self {
            Self {
                part_type: StreamPartType::ToolCall,
                thinking: None,
                text: None,
                end: None,
                tool_call: Some(StreamToolCallPart {
                    response_id: value.response_id,
                    tool_call_id: value.tool_call_id,
                    name: value.name,
                    partial_arguments: value.partial_arguments,
                }),
            }
        }
    }

    impl From<LLMStreamingComplete> for StreamPart {
        fn from(value: LLMStreamingComplete) -> Self {
            Self {
                part_type: StreamPartType::End,
                thinking: None,
                text: None,
                tool_call: None,
                end: Some(StreamEndPart {
                    id: value.id,
                    created_at: value.created_at,
                    deltas: value
                        .deltas
                        .iter()
                        .map(|s| StreamTextPart {
                            text_delta: s.delta.clone(),
                            response_id: s.response_id.clone(),
                            stop: s.stop,
                            created_at: s.created_at,
                        })
                        .collect(),
                    thinking_deltas: value.thinking_deltas.map(|v| {
                        v.iter()
                            .map(|t| StreamThinkingPart {
                                thinking_delta: t.delta.clone(),
                                response_id: t.response_id.clone(),
                                created_at: t.created_at,
                            })
                            .collect()
                    }),
                    usage: value.usage.map(LLMUsage::from),
                    message: value.message.into(),
                }),
            }
        }
    }

    impl From<LLMStreamingResponse> for StreamPart {
        fn from(value: LLMStreamingResponse) -> Self {
            match value {
                LLMStreamingResponse::Delta(d) => Self::from(d),
                LLMStreamingResponse::ToolDelta(d) => Self::from(d),
                LLMStreamingResponse::ThinkingDelta(d) => Self::from(d),
                LLMStreamingResponse::Complete(c) => Self::from(c),
            }
        }
    }

    pub type InnerPyStream = Arc<
        Mutex<
            BoxStream<
                'static,
                Result<LLMStreamingResponse, Box<dyn std::error::Error + Send + Sync>>,
            >,
        >,
    >;

    /// Async iterator that yields :py:class:`StreamPart` objects.
    ///
    /// Created by :py:meth:`LLM.stream_response`.  Use ``async for`` to consume::
    ///
    /// ```python
    ///     async for part in llm.stream_response(request):
    ///         if part.type == StreamPartType.Text:
    ///             print(part.text.text_delta)
    /// ```
    #[gen_stub_pyclass]
    #[pyclass]
    pub struct PyStream {
        inner: InnerPyStream,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyStream {
        /// Return ``self`` as the async iterator.
        fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        /// Yield the next :py:class:`StreamPart` or raise ``StopAsyncIteration``.
        fn __anext__<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
            let inner = self.inner.clone();
            pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let mut guard = inner.lock().await;
                match guard.next().await {
                    Some(Ok(item)) => Ok(StreamPart::from(item)),
                    Some(Err(e)) => Err(PyRuntimeError::new_err(format!(
                        "Streaming was interrupted due to the following error: {}",
                        e
                    ))),
                    None => Err(PyStopAsyncIteration::new_err(())),
                }
            })
        }
    }

    /// Retry policy for failed LLM requests.
    #[gen_stub_pyclass]
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

    #[gen_stub_pymethods]
    #[pymethods]
    impl RetryPolicy {
        /// Create a retry policy.
        ///
        /// Args:
        ///     max_retries: How many times to retry before giving up (default 3).
        ///     min_retry_interval: Minimum wait between retries in ms (default 500).
        ///     max_retry_interval: Maximum wait between retries in ms (default 3000).
        ///     base: Exponential-backoff base (default 2).
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

    /// LLM client.
    ///
    /// Create an instance and call :py:meth:`respond` for blocking-style
    /// interaction or :py:meth:`stream_response` for async streaming.
    #[gen_stub_pyclass]
    #[pyclass(from_py_object)]
    #[derive(Default, Clone)]
    #[allow(clippy::upper_case_acronyms)]
    pub struct LLM {
        inner: NativeLLM,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl LLM {
        /// Create a new LLM client.
        ///
        /// Args:
        ///     retry_policy: Optional :py:class:`RetryPolicy` for failed requests.
        #[new]
        #[pyo3(signature = (retry_policy = None))]
        fn new(retry_policy: Option<RetryPolicy>) -> Self {
            Self {
                inner: NativeLLM::new(retry_policy.unwrap_or_default().into()),
            }
        }

        /// Send a request and await the complete response.
        ///
        /// Args:
        ///     request: The :py:class:`LLMRequest` to send.
        ///
        /// Returns:
        ///     A coroutine that resolves to an :py:class:`LLMResponse`.
        fn respond<'py>(
            &self,
            py: Python<'py>,
            request: LLMRequest,
        ) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.inner;
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

        /// Send a request and return an async iterator over streaming parts.
        ///
        /// Args:
        ///     request: The :py:class:`LLMRequest` to send (``stream`` should be ``True``).
        ///
        /// Returns:
        ///     A :py:class:`PyStream` async iterator yielding :py:class:`StreamPart` objects.
        fn stream_response(&self, request: LLMRequest) -> PyResult<PyStream> {
            let inner = self.inner;
            let request = request.into();

            let fut = async move { inner.stream_response(request).await };

            let stream = futures::stream::once(fut).try_flatten();

            Ok(PyStream {
                inner: Arc::new(Mutex::new(Box::pin(stream))),
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn llm_usage_to_dict_preserves_other_tokens() {
            Python::initialize();
            let other_tokens = HashMap::from([("reasoning".to_string(), 2)]);
            let usage = LLMUsage::new(1, 3, None, None, Some(other_tokens.clone()));

            Python::attach(|py| {
                let usage_dict = usage.to_dict(py).unwrap();
                let actual = usage_dict
                    .get_item("other_tokens")
                    .unwrap()
                    .unwrap()
                    .extract::<Option<HashMap<String, u32>>>()
                    .unwrap();
                assert_eq!(actual, Some(other_tokens));
            });
        }

        #[test]
        fn file_urls_load_image_and_document_parts() {
            Python::initialize();
            let files = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../llms-sdk/files");
            let image_url = Url::from_file_path(files.join("cat.jpeg"))
                .unwrap()
                .to_string();
            let document_url = Url::from_file_path(files.join("file.pdf"))
                .unwrap()
                .to_string();

            Python::attach(|py| {
                assert!(matches!(
                    image_part(py, Either::Left(image_url)),
                    Ok(MessagePart::ImagePart {
                        is_base64: true,
                        ..
                    })
                ));
                assert!(matches!(
                    document_part(py, Either::Left(document_url)),
                    Ok(MessagePart::DocumentPart {
                        is_base64: true,
                        ..
                    })
                ));
            });
        }
    }
}

define_stub_info_gatherer!(stub_info);
