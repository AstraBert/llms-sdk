import asyncio
import os
import sys

from llms_sdk_py import LLM, LLMRequest, Message, TextPart


async def run(prompt: str, stream: bool) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("No OPENAI_API_KEY in the environment")
    text = TextPart(prompt)
    message = Message("user", [text])
    req = LLMRequest.from_defaults(api_key, [message], "gpt-5.4-nano", stream = stream)
    llm = LLM()
    if stream:
        async for part in llm.stream_response(req):
            print(part.to_dict())
    else:
        response = await llm.respond(req)
        print(response.to_dict())

def main() -> None:
    args = sys.argv
    if len(args) < 2:
        print("A prompt is required as a positional arguments")
    prompt = args[1]
    stream = False
    if len(args) >= 3:
        stream = args[2] == "--stream"
    asyncio.run(run(prompt, stream))

if __name__ == "__main__":
    main()
