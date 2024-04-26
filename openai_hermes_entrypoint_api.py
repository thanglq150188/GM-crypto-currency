import torch
import json

from transformers import AutoTokenizer
from rag import config

from prompter import PromptManager

from utils import (
    inference_logger,
    get_chat_template,
    validate_and_extract_tool_calls,
    transform_dict
)

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

import uuid
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
# openai_api_base = "https://gm-ai-model-api-dev.uslab.dev/v1"
openai_api_base = "http://localhost:8000/v1"
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
    
    tools = [transform_dict(tool) for tool in tools]
    
    for tool in tools:
        print(tools)
    
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
    return json.loads(json.dumps(completion, default=lambda o: o.__dict__, indent=2))

app = FastAPI()

@app.post("/")
async def receive_data(request: Request):
    data = await request.json()
    
    response = await function_calling_router(data)
    
    assistant_message = response['choices'][0]['text']
    
    validation_result, tool_calls, error_message = validate_and_extract_tool_calls(assistant_message)
    
    if len(tool_calls) > 0 :
        response['choices'] = [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": None,
                    "role": "assistant",
                    "function_call": None,
                    "tool_calls": []
                }
            }
        ]
        for tool in tool_calls:
            response['choices'][0]['message']['tool_calls'].append(
                {
                    'id': f'call_{uuid.uuid4()}',
                    'function': {
                        "name": tool['name'],
                        "argument": json.dumps(tool['arguments'])
                    },
                    'type': 'function'
                }
            )
    
    return JSONResponse(response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)