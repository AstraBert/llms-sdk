mod types;

pub use types::*;

use futures::channel::oneshot;
use futures_util::StreamExt;
use js_sys::Function;
use llms_sdk::LLM;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;

#[wasm_bindgen(start)]
pub fn __wasm_start() {
    console_error_panic_hook::set_once();
}

/// Send a single-turn (non-streaming) chat request and return the full response.
#[wasm_bindgen]
pub async fn chat(request: LLMRequest) -> Result<LLMResponse, JsError> {
    let llm = LLM::default();
    let response = llm
        .respond(request.try_into()?)
        .await
        .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(response.into())
}

/// Send a streaming chat request, invoking `callback(error, chunk)` for every event.
///
/// The callback receives two arguments:
/// - `error` – `null` on success or a string message on failure.
/// - `chunk` – an [`LLMStreamingResponse`] variant on success, or `undefined` on failure.
#[wasm_bindgen(js_name = streamChat)]
pub async fn stream_chat(request: LLMRequest, callback: Function) -> Result<(), JsError> {
    let llm = LLM::default();
    let mut stream = llm
        .stream_response(request.try_into()?)
        .await
        .map_err(|e| JsError::new(&format!("Failed to produce stream: {}", e)))?;

    let (tx, rx) = oneshot::channel::<()>();

    spawn_local(async move {
        let this = JsValue::NULL;
        while let Some(item) = stream.next().await {
            match item {
                Ok(response) => {
                    match serde_wasm_bindgen::to_value(&LLMStreamingResponse::from(response)) {
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
                    }
                }
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
        let _ = tx.send(());
    });

    rx.await
        .map_err(|e| JsError::new(&format!("Stream task dropped before completion: {}", e)))?;

    Ok(())
}
