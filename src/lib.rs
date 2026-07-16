pub mod anthropic;
pub mod errors;
pub mod openai;
mod types;

pub use types::*;

use crate::{anthropic::AntClient, openai::OpenAIClient};

#[derive(Debug, Default)]
pub struct LLM {
    pub retry_policy: RetryPolicy,
}

impl LLM {
    pub fn new(retry_policy: RetryPolicy) -> Self {
        Self { retry_policy }
    }
    pub async fn respond(
        &self,
        request: LLMRequest,
    ) -> Result<LLMResponse, Box<dyn std::error::Error>> {
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
    pub async fn stream_response(
        &self,
        request: LLMRequest,
    ) -> Result<LLMStream, Box<dyn std::error::Error>> {
        let api_type = request.api_type;
        match api_type {
            ApiType::OpenAI => {
                let openai_client = OpenAIClient::new(self.retry_policy);
                let stream = openai_client.stream_response(request).await?;
                Ok(stream)
            }
            ApiType::Anthropic => Err("Not yet implemented".into()),
        }
    }
}
