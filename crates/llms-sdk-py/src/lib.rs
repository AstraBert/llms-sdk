use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod llms_sdk {
    use std::fs;

    use base64::prelude::*;
    use either::Either;
    use llms_sdk::{ALLOWED_IMAGE_TYPES, AudioPart};
    use pyo3::{
        exceptions::{PyAttributeError, PyKeyError, PyValueError},
        prelude::*,
        types::PyDict,
    };
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
        fn part_type(&self) -> String {
            match self {
                Self::TextPart {
                    r#type: tp,
                    text: _,
                } => tp.__str__(),
                Self::ThinkingPart {
                    r#type: tp,
                    thinking: _,
                    signature: _,
                } => tp.__str__(),
                Self::ToolCallPart {
                    r#type: tp,
                    id: _,
                    name: _,
                    arguments: _,
                } => tp.__str__(),
                Self::ToolResultPart {
                    r#type: tp,
                    tool_call_id: _,
                    result: _,
                } => tp.__str__(),
                Self::ImagePart {
                    r#type: tp,
                    image_data: _,
                    is_base64: _,
                    mime_type: _,
                } => tp.__str__(),
                Self::DocumentPart {
                    r#type: tp,
                    document_data: _,
                    is_base64: _,
                    mime_type: _,
                } => tp.__str__(),
                Self::AudioPart {
                    r#type: tp,
                    audio_data: _,
                    mime_type: _,
                } => tp.__str__(),
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
}
