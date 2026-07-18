use std::fmt::Display;

/// Error returned when a string cannot be parsed into a supported enum variant.
#[derive(Debug)]
pub struct UnsupportedType {
    pub unsupported_type: String,
}

impl Display for UnsupportedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unsupported enum type: {}", self.unsupported_type)
    }
}

impl std::error::Error for UnsupportedType {}

/// Error returned when input data fails validation.
#[derive(Debug)]
pub struct InvalidInput {
    pub reason: String,
}

impl Display for InvalidInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid input data: {}", self.reason)
    }
}

impl std::error::Error for InvalidInput {}

/// Error returned when a message part type is not supported for the target API.
#[derive(Debug)]
pub struct UnsupportedPartType {
    pub part_type: String,
    pub api_type: String,
}

impl Display for UnsupportedPartType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unsupported part type ({}) for API {}",
            self.part_type, self.api_type
        )
    }
}

impl std::error::Error for UnsupportedPartType {}

/// Error returned when the `stream` flag on a request conflicts with the called method.
#[derive(Debug)]
pub struct StreamParamError {
    pub should_stream: bool,
}

impl std::error::Error for StreamParamError {}

impl Display for StreamParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = if self.should_stream {
            "stream is set to False, but a method that implies streaming has been called"
        } else {
            "stream is set to True, but a method that does not imply streaming has been called"
        };
        write!(f, "{}", message)
    }
}

/// Error returned when a prompt cache TTL value is not supported.
#[derive(Debug)]
pub struct InvalidTtl {
    pub ttl: String,
}

impl std::error::Error for InvalidTtl {}

impl Display for InvalidTtl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "The requested TTL is not supported: {}", self.ttl)
    }
}

/// Error returned when an [`LLMRequest`](crate::LLMRequest) cannot be converted into an Anthropic request.
#[derive(Debug)]
pub struct InvalidAntRequestConversion {
    pub reason: String,
}

impl std::error::Error for InvalidAntRequestConversion {}

impl Display for InvalidAntRequestConversion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Unable to convert request to Anthropic request: {}",
            self.reason
        )
    }
}

impl From<InvalidTtl> for InvalidAntRequestConversion {
    fn from(value: InvalidTtl) -> Self {
        Self {
            reason: format!("{}", value),
        }
    }
}

impl From<UnsupportedPartType> for InvalidAntRequestConversion {
    fn from(value: UnsupportedPartType) -> Self {
        Self {
            reason: format!("{}", value),
        }
    }
}
