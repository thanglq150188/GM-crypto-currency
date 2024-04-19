import re
import inspect
import requests
import pandas as pd
import yfinance as yf
import concurrent.futures

from typing import List
from bs4 import BeautifulSoup
from utils import inference_logger
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool



@tool
def get_portfolio(telegram_id: str) -> dict | str:
    """
    Get portfolio/balance of the user. Before performing any action such as swapping, bridging, or transferring, you must call this function to get the exact user balance for calculations.",

    Args:
        telegram_id (str): The telegram id of user.

    Returns:
        dict | str: A dictionary containing variables declared and values returned by function calls,
            or an error message if an exception occurred.
    """
    pass

@tool
def swap_token(input_token_symbol, output_token_symbol, telegram_id,
               input_token_amount=None, output_token_amount=None,
               output_token_address=None,
               slippage=0.01) -> dict:
    """
    Useful when need to buy/sell/swap.
            

    Args:
        input_token_amount (float): The input token amount.
        output_token_amount (float): The output token amount.
        input_token_symbol (str): The input token symbol
        output_token_symbol (str): The output token symbol.
        output_token_address (str): The output token address
        slippage (str): The slippage to swap (percent)
        telegram_id (str): The telegram id of user.
    """
    pass

@tool
def tokens_supported(chain_name: str, token_address: str, token_symbol: str) -> float:
  """
  Get the list of tokens that are currently supported in the system

  Args:
    chain_name (str): The chain name.
    token_address (str): The token address.
    token_symbol (str): The token symbol.

  Returns:
    List: List of tokens that are currently supported in the system
  """
  pass

@tool
def get_market_info_token(token_symbol: str, action_type: str, interval: str) -> dict:
    """
    Price, Market cap, and volume 24h of cryptocurrency

    Args:
        token_symbol (str): The input token symbol.
        action_type (str): The action type from input. MUST be one of the following: ['Market cap', 'Volume', 'Price', 'FDV']
        interval (str): The interval time (e.g., 7 days, 1 week)
    """
    pass

@tool
def transfer_token(telegram_id: str, token_address: str, amount: str,
                   token_symbol: str, chain_name: str, destination_wallet_address: str) -> dict:
    """
    Transfer token from your wallet to another using token address, token symbol, destination wallet address, chain name, amount with telegram id. Return the information that needs to be transferred. You must call 'get_portfolio' function to get exact user balance and calculate before transfer

    Args:
        telegram_id (str): The telegram id of user.
        token_address (str): The input token address.
        amount (str): The amount of token needed to be transfered.
        token_symbol (str): the input token symbol.
        chain_name (str): the chain name to transfer.
        destination_wallet_address (str): The destination wallet address.
    """
    pass

@tool
def get_document(input: str) -> dict:
    """
    Information about project documents, including: Intent Trade (Intent), Whales Market (Whales), Solana, Jupiter. If you have any questions about documenting your projects, you should use this tool!

    Args:
        input (str): The input for getting documents.
    """
    pass

@tool
def get_percent_balance(input_token_address: str, percent: float, chain_name: str) -> pd.DataFrame:
    """
    Calculate percent of token balance.

    Args:
        input_token_address (str): The input token address.
        percent (float): The percent of token balance.
        chain_name (str): Chain name - balance on this chain.
    """
    pass

def get_openai_tools() -> List[dict]:
    functions = [
        get_portfolio,
        swap_token,
        tokens_supported,
        get_market_info_token,
        transfer_token,
        get_document,
        get_percent_balance
    ]

    tools = [convert_to_openai_tool(f) for f in functions]
    return tools