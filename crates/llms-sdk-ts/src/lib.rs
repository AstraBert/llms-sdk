#![deny(clippy::all)]

mod types;

use std::time::Duration;

use llms_sdk::RetryPolicy as NativeRetryPolicy;
use llms_sdk::LLM as NativeLLM;
use napi_derive::napi;
pub use types::*;

#[napi]
pub struct LLM {
  inner: NativeLLM,
}

#[napi(object)]
pub struct RetryPolicy {
  pub max_retries: u32,
  pub min_retry_interval: u32,
  pub max_retry_interval: u32,
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

impl Default for LLM {
  fn default() -> Self {
    Self {
      inner: NativeLLM::default(),
    }
  }
}

#[napi]
impl LLM {
  #[napi(constructor)]
  pub fn new(retry_policy: Option<RetryPolicy>) -> Self {
    Self {
      inner: NativeLLM::new(NativeRetryPolicy::from(retry_policy.unwrap_or_default())),
    }
  }
}
