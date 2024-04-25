import torch
import json

from transformers import AutoTokenizer

from prompter import PromptManager

from utils import (
    inference_logger,
    get_chat_template,
    validate_and_extract_tool_calls
)

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "https://gm-ai-model-api-dev.uslab.dev/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

prompter = PromptManager()

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

if tokenizer.chat_template is None:
    print("No chat template defined, getting chat_template...")
    tokenizer.chat_template = get_chat_template(chat_template)

async def function_calling_router(data):
    tools = data.get("tools")
    messages = data.get("messages")
    
    prompt = prompter.generate_prompt(messages, tools, num_fewshot=None)
    
    template_prompt = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=False
    )
    
    inference_logger.info(template_prompt)
    
    completion = client.completions.create(
        model=data.get("model"),
        prompt=template_prompt,
        max_tokens=1500,
        top_p=data.get("top_p"),
        temperature=data.get("temperature"),
        frequency_penalty=data.get("frequency_penalty"),
        presence_penalty=data.get("presence_penalty"),
        n=data.get("n"),
        stream=False
    )    
    return json.dumps(completion, default=lambda o: o.__dict__, indent=2)

app = FastAPI()

@app.post("/")
async def receive_data(request: Request):
    data = await request.json()
    
    response = await function_calling_router(data)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)