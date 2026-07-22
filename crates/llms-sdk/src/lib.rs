//! Unified interface to call OpenAI and Anthropic-compatible LLM APIs from Rust.
//!
//! The main entry point is [`LLM`], which dispatches requests to the appropriate
//! provider based on [`ApiType`].

pub mod anthropic;
pub mod errors;
pub mod openai;
mod types;

pub use types::*;

use rustls::crypto::ring::default_provider;

/// Installs the `ring` crypto provider for `rustls`.
///
/// This should be called once before making any HTTPS requests.
/// It is safe to call multiple times (subsequent calls are no-ops).
pub fn install_crypto_provider() {
    let _ = default_provider().install_default();
}

use crate::{anthropic::AntClient, openai::OpenAIClient};

/// Unified entry point for sending requests to OpenAI or Anthropic-compatible APIs.
#[derive(Debug, Default)]
pub struct LLM {
    /// Retry policy applied to all API requests made through this client.
    pub retry_policy: RetryPolicy,
}

impl LLM {
    /// Creates a new [`LLM`] client with the provided retry policy.
    pub fn new(retry_policy: RetryPolicy) -> Self {
        Self { retry_policy }
    }

    /// Sends a single completion request and returns the full response.
    pub async fn respond(
        &self,
        request: LLMRequest,
    ) -> Result<LLMResponse, Box<dyn std::error::Error>> {
        install_crypto_provider();
        let api_type = request.api_type;
        match api_type {
            ApiType::OpenAI => {
                let openai_client = OpenAIClient::new(self.retry_policy);
                let response = openai_client.respond(request).await?;
                Ok(response)
            }
            ApiType::Anthropic => {
                let ant_client = AntClient::new(self.retry_policy);
                let response = ant_client.respond(request).await?;
                Ok(response)
            }
        }
    }

    /// Sends a streaming completion request and returns a stream of partial responses.
    pub async fn stream_response(
        &self,
        request: LLMRequest,
    ) -> Result<LLMStream, Box<dyn std::error::Error>> {
        install_crypto_provider();
        let api_type = request.api_type;
        match api_type {
            ApiType::OpenAI => {
                let openai_client = OpenAIClient::new(self.retry_policy);
                let stream = openai_client.stream_response(request).await?;
                Ok(stream)
            }
            ApiType::Anthropic => {
                let ant_client = AntClient::new(self.retry_policy);
                let stream = ant_client.stream_response(request).await?;
                Ok(stream)
            }
        }
    }
}
