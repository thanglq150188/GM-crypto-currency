import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_fireworks import ChatFireworks
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, FunctionMessage
import openai
import json
from pathlib import Path
from pprint import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
import emb_utils
import config


intent_trade_instructions = """As Intent.Trade, your role is to facilitate cryptocurrency trading and on-chain interactions within the Solana network, while also being integrated with "GM AI". You specialize in trading cryptocurrencies and can perform various on-chain interactions. Your greetings should reflect your role and capabilities, providing a friendly and welcoming introduction to users. However, limit the use of emojis to a maximum of two times in a response.

Your current on-chain actions include token price checks, user wallet balance checks, token swapping, and token transfers. It's important to note that the platform fee is 1%% of the transaction amount, and this information should be disclosed only upon direct user inquiry. Additionally, it's essential to specify that the platform is tailored for blockchain-savvy individuals, cryptocurrency traders, crypto enthusiasts, and finance and trading professionals. You must also emphasize that you can communicate in multiple languages and respond to questions in the language they are asked in.

When responding to balance inquiries, use the following format to display the balance in ascending order: check balance ({"💳 Network 1 : ": \n Element in Set 1 \n Element in Set 2 ------------------------------------------ 💳 Network 2 : }). Additionally, if a user wants to check their wallet address, direct them to use the '/wallets' command to manage their wallet and provide the necessary information.
If the queries relate to price, Fully Diluted Valuation (FDV), market cap, or 24-hour trading volume you must call the 'get_market_info_token' function before answer.

Before performing any action such as swapping, bridging, or transferring, prompt the user for all parameters for a function call and call the 'get-portfolio-tools' function to acquire the exact user balance for calculations.

For token swapping, ask the user for the input token symbol (default: ""), output token symbol (default: ""), input token amount or output token amount (at least one), and slippage (default: 1%).  Don't run price-tool and Token Available while swapping.
Following tips to check the input and output tokens/amounts
- Buy 1000  A with  B: output token amount = 1000, output token symbol = A, input token symbol = B
- Buy  A with 1000  B: input token amount = 1000, input token symbol = B, output token symbol = A
- Sell 1000  A for  B: input token amount = 1000, input token symbol = A, output token symbol = B
- Sell A for 1000  B: output token amount = 1000, output token symbol = B, input token symbol = A
- Swap 30% my A balance to B on chain: input token amount = 30%% balance of token A
- Buy 1000 A B: input token amount = 1000, output token symbol =  B, input token symbol =  A

In the case of token transfers, ask the user for the token (address), destination wallet address, amount, and the action "transfer".
If there's no relevant answer available, respond with "Sorry, I'm not familiar with the information you just provided. I'll update you as soon as it's within my knowledge base." Remember to keep the response conversational and familiar with Telegram, and refrain from including Telegram ID, reference links, sources in the answer."""

intent_trade_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": "Get portfolio/balance of the user. Before performing any action such as swapping, bridging, or transferring, you must call this function to get the exact user balance for calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "telegram_id": {
                        "type": "string",
                        "description": "The telegram id of user. Default is '000'"
                    }
                },
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#",
                "required": [
                    "telegram_id"
                ],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "swap_token",
            "description": "Useful when need to buy/sell/swap. You must call get-portfolio function to get exact user balance and calculate before swap",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_token_amount": {
                        "type": ["number", "null"],
                        "description": "The input token amount"
                    },
                    "output_token_amount": {
                        "type": ["number", "null"],
                        "description": "The output token amount"
                    },
                    "input_token_symbol": {
                        "type": "string",
                        "description": "The input token symbol"
                    },
                    "output_token_symbol": {
                        "type": "string",
                        "description": "The output token symbol. Default is empty string"
                    },
                    "output_token_address": {
                        "type": "string",
                        "description": "The output token address"
                    },
                    "slippage": {
                        "type": "string",
                        "description": "The slippage to swap (percent)."
                    },
                    "telegram_id": {
                        "type": ["string", "number"],
                        "description": "The telegram id of user. Default is empty string"
                    }
                },
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tokens_supported",
            "description": "List of tokens that are currently supported in the system",
            "parameters": {
                "type": "object",
                "properties": {
                    "chain_name": {
                        "type": "string",
                        "description": "The chain name"
                    },
                    "token_address": {
                        "type": "string",
                        "description": "The token address"
                    },
                    "token_symbol": {
                        "type": "string",
                        "description": "The token symbol"
                    }
                },
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_info_token",
            "description": "Price, Market cap, and volume 24h of cryptocurrency",
            "parameters": {
                "type": "object",
                "properties": {
                    "token_symbol": {
                        "type": "string",
                        "description": "The input token symbol"
                    },
                    "action_type": {
                        "type": "string",
                        "enum": ["Market cap", "Volume", "Price", "FDV"],
                        "description": "The action type from input"
                    },
                    "interval": {
                        "type": "string",
                        "description": "The interval time (e.g., 7 days, 1 week)"
                    }
                },
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_token",
            "description": "Transfer token from your wallet to another using token address, token symbol, destination wallet address, chain name, amount with telegram id. Return the information that needs to be transferred. You must call 'get_portfolio' function to get exact user balance and calculate before transfer",
            "parameters": {
                "type": "object",
                "properties": {
                    "telegram_id": {
                        "type":"string",
                        "description": "The telegram id of user"
                    },
                    "token_address": {
                        "type": "string",
                        "description": "The token address"
                    },
                    "amount": {
                        "type": "number",
                        "description": "The token amount"
                    },
                    "token_symbol": {
                        "type": "string",
                        "description": "The token symbol"
                    },
                    "chain_name": {
                        "type": "string",
                        "description": "The chain name to transfer"
                    },
                    "destination_wallet_address": {
                        "type": "string",
                        "description": "The destination wallet address"
                    }
                },
                "required": [
                    "telegram_id",
                    "token_address",
                    "amount",
                    "token_symbol",
                    "chain_name",
                    "destination_wallet_address"
                ],
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Information about project documents, including: Intent Trade (Intent), Whales Market (Whales), Solana, Jupiter. If you have any questions about documenting your projects, you should use this tool!",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string"
                    }
                },
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_percent_balance",
            "description": "Calculate percent of token balance",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_token_address": {
                        "type": "string",
                        "description": "The input token address"
                    },
                    "percent": {
                        "type": "number",
                        "description": "The percent of token balance"
                    },
                    "chain_name": {
                        "type": "string",
                        "description": "Chain name - balance on this chain"
                    }
                },
                "required": [
                    "input_token_address",
                    "percent"
                ],
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    }
]

another_us_instructions = """Instructions on how to use the get-followers-twitter tool:
    1. Using the tool:
    - This tool will be run after the user enters a question which can contain any user's username. If this username has not found or user has not ever inputted then it will return the followers of Elon Musk (CEO Twitter)
    Example:
        - User: Get the followers on twitter of username elonmusk
        - Assistant: Here are some followers of username elomusk:
          - MemezCherub (https://twitter.com/godofmemez)
          - Candi Stelman (http://tinyurl.com/22ca4rst)
          - Tracee Ramire (http://tinyurl.com/28wum4rl)
          - Keiko Beckelhimer (http://tinyurl.com/2ddcqqd9)"""

another_us_tools = [
    {
        "type": "function",
        "function": {
            "name": "get-followers-twitter",
            "description": "Using this method you can get the followers of a user on Twitter. You must call the get-followers-twitter tool to get the followers of a user",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Twitter user's name"
                    }
                },
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get-posts-twitter",
            "description": "Using this method you can get the posts of a user on Twitter. You must call the get-posts-twitter tool to get the user's posts",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Twitter user's name"
                    },
                    "rest_id": {
                        "type": "string",
                        "description": "Twitter user's id. This parameter overwrites the username parameter."
                    }
                },
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get-user-twitter-info",
            "description": "Using this method you can get information about a user on Twitter. You must call the get-user-twitter-info tool to get the Twitter user's information",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Twitter user's name"
                    },
                    "rest_id": {
                        "type": "string",
                        "description": "Twitter user's id. This parameter overwrites the username parameter."
                    }
                },
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_id_coin_gecko",
            "description": "This method provides a list of tokens that need to get information on CoinGecko based on the name or symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string"
                    }
                },
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_history_coin_gecko",
            "description": "Get historical data at a given date for a coin, including: id, symbol, image, developer_data, community_data, market data containing data of price, market cap, total volume",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The CoinGecko id of the token (can be obtained from /coins) e.g., bitcoin"
                    },
                    "date": {
                        "type": "string",
                        "description": "The date of the data snapshot in dd-mm-yyyy format e.g., 30-12-2022"
                    },
                    "localization": {
                        "type": "boolean",
                        "description": "Set to False to exclude localized languages in the response"
                    }
                },
                "required": ["id", "date", "localization"],
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_price_coin_gecko",
            "description": "Get the price of the cryptocurrency, including: id, symbol, name, image, current price, market cap, market cap rank, fully diluted valuation, total volume, high 24h, low 24h, price change 24h, price change percentage 24h, market cap change 24h, market cap change percentage 24h, circulating supply, total supply, max supply, ath, ath change percentage, ath date,atl, atl change percentage, atl date, roi, last updated",
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "string",
                        "description": "The ids of the coins, comma-separated cryptocurrency symbols (base). e.g., 'bitcoin,ethereum'"
                    },
                    "vs_currency": {
                        "type": "string",
                        "description": "The target currency of market data (USD, EUR, JPY, etc.)"
                    }
                },
                "required": ["ids", "vs_currency"],
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_trending_coin_gecko",
            "description": "Get trending search coins (top 7) on CoinGecko in the last 24 hours",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string"
                    }
                },
                "additionalProperties": False,
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    }
]

# Setting additional parameters: temperature, max_tokens, top_p
fw_dbrx = ChatFireworks(
    model="accounts/fireworks/models/dbrx-instruct",
    temperature=0,
    max_tokens=1000,
    fireworks_api_key="If28EuWQZJF34b2FR8VG3N5WRdWj7QasZZgJ1TP58vvCfxks"
)



def query_refinement(history, message):
    REFINEMENT_PROMPT = """
Please rewrite the user's conversation into a single question, focusing on the main point being asked at the end and removing any extraneous information.

Conversation:
{conversation}

Your response:"""

    conversation = ""
    for human, assistant in history:
        conversation += f"Customer: {human}\n"
        conversation += f"Assistant: {assistant}\n"
    
    conversation += f"Customer: {message}\n"

    human_message = HumanMessage(content=REFINEMENT_PROMPT.format(conversation=conversation))

    response = fw_dbrx.invoke([human_message])

    return response.content

def qa_with_rag(collection_name, message, history, sample_answer=None):
    
    RAG_PROMPT = """Your task is to answer the user's question based on the provided data.
If it is a general greetings question, simply provide a polite response.
If the given data is not relevant to the question, please respond with "I don't know". Don't try to make up an answer.
{sample}
Data: {context}
Question: {question}
Answer:"""

    refinement = True
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append(HumanMessage(content=human))
        history_openai_format.append(AIMessage(content=assistant))

    if refinement:
      query = query_refinement(history, message)
    else:
      query = message

    docs = emb_utils.similar_search(collection_name=collection_name, query=query)

    context = '\n---------------\n'.join([doc.page_content for doc in docs])
    if sample_answer is None:
      history_openai_format.append({"role": "user", "content": RAG_PROMPT.format(context=context, question=message, sample='')})
    else:
      history_openai_format.append({"role": "user", "content": RAG_PROMPT.format(context=context, question=message, sample=f'Sample Answer: {sample_answer}')})
    for i in range(10):
        try:
            return fw_dbrx.invoke(history_openai_format)
        except Exception as ex:
            print(ex)
            time.sleep(3)

    return None

def function_calling_chat(instruction, histories, query, tools):
    client = openai.OpenAI(
        base_url = "https://api.fireworks.ai/inference/v1",
        api_key = "If28EuWQZJF34b2FR8VG3N5WRdWj7QasZZgJ1TP58vvCfxks"
    )

    messages = [
        {"role": "system", "content": instruction},
    ]

    for human, assistant in histories:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": query})

    chat_completion = client.chat.completions.create(
        model="accounts/fireworks/models/firefunction-v1",
        messages=messages,
        tools=tools,
        temperature=0.0
    )

    return chat_completion
