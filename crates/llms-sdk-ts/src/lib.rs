#![deny(clippy::all)]

mod types;

use futures_util::StreamExt;
use std::time::Duration;

use llms_sdk::RetryPolicy as NativeRetryPolicy;
use llms_sdk::LLM as NativeLLM;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
pub use types::*;

/// LLM client.
///
/// Create an instance with `new LLM(retryPolicy?)` and call
/// `respond()` for a one-shot completion or `streamResponse()`
/// for streaming output via a callback.
#[napi]
#[derive(Default)]
pub struct LLM {
  inner: NativeLLM,
}

/// Retry configuration used when constructing an [`LLM`].
#[napi(object)]
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

#[napi]
impl LLM {
  /// Create a new LLM client.
  ///
  /// @param retryPolicy - Optional retry policy. Defaults to 3 retries with
  ///   exponential backoff between 500 ms and 3000 ms.
  #[napi(constructor)]
  pub fn new(retry_policy: Option<RetryPolicy>) -> Self {
    Self {
      inner: NativeLLM::new(NativeRetryPolicy::from(retry_policy.unwrap_or_default())),
    }
  }

  /// Send a single request and wait for the full response.
  ///
  /// @param request - The [`LLMRequest`] to send.
  /// @returns A complete [`LLMResponse`] containing the generated message and usage stats.
  #[napi]
  pub async fn respond(&self, request: LLMRequest) -> napi::Result<LLMResponse> {
    let response = self.inner.respond(request.try_into()?).await.map_err(|e| {
      napi::Error::new(
        napi::Status::GenericFailure,
        format!("Failed to produce response: {}", e),
      )
    })?;
    Ok(response.into())
  }

  /// Stream a response incrementally through a JavaScript callback.
  ///
  /// The callback is invoked for every chunk (`LLMStreamingResponse`) as it
  /// arrives from the provider. The final chunk is always a `Complete` variant.
  ///
  /// @param request - The [`LLMRequest`] to send.
  /// @param callback - Node-style callback receiving `(err, chunk)`.
  #[napi]
  pub async fn stream_response(
    &self,
    request: LLMRequest,
    #[napi(ts_arg_type = "(err: Error | null, chunk?: LLMStreamingResponse) => void")]
    callback: ThreadsafeFunction<LLMStreamingResponse, ()>,
  ) -> napi::Result<()> {
    let mut stream = self
      .inner
      .stream_response(request.try_into()?)
      .await
      .map_err(|e| {
        napi::Error::new(
          napi::Status::GenericFailure,
          format!("Failed to produce stream: {}", e),
        )
      })?;
    napi::tokio::spawn(async move {
      while let Some(item) = stream.next().await {
        match item {
          Ok(response) => {
            callback.call(Ok(response.into()), ThreadsafeFunctionCallMode::NonBlocking);
          }
          Err(e) => {
            let err = napi::Error::new(napi::Status::GenericFailure, e.to_string());
            callback.call(Err(err), ThreadsafeFunctionCallMode::NonBlocking);
            break;
          }
        }
      }
    });

    Ok(())
  }
}
