mod types;

use std::time::Duration;

use futures_util::StreamExt;
use js_sys::Function;
use llms_sdk::LLM;
use llms_sdk::RetryPolicy as NativeRetryPolicy;
use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
pub use types::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;

#[derive(Debug, Serialize, Deserialize, Clone, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct RetryPolicy {
    /// Maximum number of retries before giving up.
    pub max_retries: u32,
    /// Minimum wait time between retries, in milliseconds.
    pub min_retry_interval: u32,
    /// Maximum wait time between retries, in milliseconds.
    pub max_retry_interval: u32,
    /// Exponential backoff base.
    pub base: u32,
}

impl From<RetryPolicy> for NativeRetryPolicy {
    fn from(value: RetryPolicy) -> Self {
        Self {
            max_retries: value.max_retries,
            max_retry_interval: Duration::from_millis(value.max_retry_interval as u64),
            min_retry_interval: Duration::from_millis(value.min_retry_interval as u64),
            base: value.base,
            ..Default::default()
        }
    }
}

#[wasm_bindgen]
pub async fn chat(
    request: LLMRequest,
    retry_policy: Option<RetryPolicy>,
) -> Result<LLMResponse, JsError> {
    let llm = LLM::new(retry_policy.map_or(NativeRetryPolicy::default(), NativeRetryPolicy::from));
    let response = llm
        .respond(request.try_into()?)
        .await
        .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(response.into())
}

#[wasm_bindgen(js_name = streamChat)]
pub async fn stream_chat(
    request: LLMRequest, // wasm-facing (Tsify) request type
    callback: Function,
    retry_policy: Option<RetryPolicy>,
) -> Result<(), JsError> {
    let llm = LLM::new(retry_policy.map_or(NativeRetryPolicy::default(), NativeRetryPolicy::from));
    let mut stream = llm
        .stream_response(request.try_into()?)
        .await
        .map_err(|e| JsError::new(&format!("Failed to produce stream: {}", e)))?;

    spawn_local(async move {
        let this = JsValue::NULL;
        while let Some(item) = stream.next().await {
            match item {
                Ok(response) => match serde_wasm_bindgen::to_value(&response) {
                    Ok(chunk) => {
                        let _ = callback.call2(&this, &JsValue::NULL, &chunk);
                    }
                    Err(e) => {
                        let _ = callback.call2(
                            &this,
                            &JsValue::from_str(&e.to_string()),
                            &JsValue::UNDEFINED,
                        );
                        break;
                    }
                },
                Err(e) => {
                    let _ = callback.call2(
                        &this,
                        &JsValue::from_str(&e.to_string()),
                        &JsValue::UNDEFINED,
                    );
                    break;
                }
            }
        }
    });

    Ok(())
}
