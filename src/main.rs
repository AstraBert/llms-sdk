use llms_sdk::{LLM, LLMRequest, Message, MessagePart, TextPart};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLM::default();
    let request = LLMRequest::builder()
        .api_type(llms_sdk::ApiType::OpenAI)
        .api_key(std::env::var("OPENAI_API_KEY").unwrap())
        .model("gpt-5.4-mini")
        .stream(false)
        .messages(vec![Message {
            role: llms_sdk::MessageRole::User,
            content: vec![MessagePart::Text(TextPart::new(
                "hello there, who are you?",
            ))],
        }])
        .build();
    let response = llm.respond(request).await?;
    dbg!(response);
    Ok(())
}
