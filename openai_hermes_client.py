import requests
from utils import inference_logger

payload = {
    "model": "llms/gm-01",
    "temperature": 0,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "stream": True,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get-weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date to get weather for"
                        },
                        "location": {
                            "type": "string",
                            "description": "Location to get weather for"
                        }
                    },
                    "required": [
                        "date",
                        "location"
                    ],
                    "additionalProperties": False,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get-balance",
                "description": "Get balance of a user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "number",
                            "description": "The id of the user"
                        }
                    },
                    "required": [
                        "user_id"
                    ],
                    "additionalProperties": False,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "swap-token",
                "description": "Swap token of user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_symbol": {
                            "type": "string",
                            "description": "The input symbol to convert"
                        },
                        "input_amount": {
                            "type": "number",
                            "description": "The input amount to convert"
                        },
                        "output_symbol": {
                            "type": "string",
                            "description": "The output symbol to convert"
                        }
                    },
                    "required": [
                        "input_symbol",
                        "input_amount",
                        "output_symbol"
                    ],
                    "additionalProperties": False,
                    "$schema": "http://json-schema.org/draft-07/schema#"
                }
            }
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": "What is bitcoin?"
        }
    ]
}

url = "http://localhost:8000"  # Replace with the appropriate URL of your FastAPI server

with requests.post(url, json=payload) as r:
    import json
    
    print(json.loads(r.content.decode()))