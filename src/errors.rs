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
