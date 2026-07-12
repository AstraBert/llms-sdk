use std::fmt::Display;

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
