import argparse
import torch
import json

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig
)

import functions as itf
from prompter import PromptManager
from validator import validate_function_call_schema

from utils import (
    print_nous_text_art,
    inference_logger,
    get_assistant_message,
    get_chat_template,
    validate_and_extract_tool_calls
)


prompter = PromptManager()

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

if tokenizer.chat_template is None:
    print("No chat template defined, getting chat_template...")
    tokenizer.chat_template = get_chat_template(chat_template)
    
chat = [{"role": "user", "content": "cuộc sống có giống cuộc đời ko"}]
tools = itf.get_openai_tools()
print('TOOLS----')
for tool in tools:
    from pprint import pprint
    pprint(tool)
    print()
print('------')
prompt = prompter.generate_prompt(chat, tools, num_fewshot=None)


template_prompt = tokenizer.apply_chat_template(
    prompt,
    add_generation_prompt=True,
    tokenize=False
)

# inference_logger.info(template_prompt)


from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

completion = client.completions.create(
    model='llms/gm-01',
    prompt=template_prompt,
    max_tokens=1500,
    temperature=0.0,
    n=1,
    stream=False
)

# for chunk in completion:
#     import json

#     # Assuming you have the ChatCompletionChunk object stored in a variable called `chunk`
#     chunk_json = json.dumps(chunk, default=lambda o: o.__dict__, indent=2)
#     inference_logger.info(chunk_json)

response_json = json.dumps(completion, default=lambda o: o.__dict__, indent=2)

inference_logger.info(response_json)


# assistant_message = completion.choices[0].text
# validation, tool_calls, error_message = validate_and_extract_tool_calls(assistant_message)

# inference_logger.info('COMPLETION:')
# inference_logger.info(f'validation: {validation}')
# inference_logger.info(f'tool_calls = {tool_calls}')
# inference_logger.info(f'error message = {error_message}')