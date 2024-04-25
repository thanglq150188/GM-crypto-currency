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
def get_balance(user_id: str) -> dict | str:
    """
    Get balance of a user.
    
    Args:
        user_id (str): The id of the user.
    """
    pass


@tool
def swap_token(input_symbol: str, input_amount: int, output_symbol: str) -> dict | str:
    """
    Swap token of user
    
    Args:
        input_symbol (str): The input symbol to convert. Default is 'UNK'
        input_amount (str): The input amount to convert. Default is '-1'
        output_symbol (str): The output symbol to convert. Default is 'UNK'
    """

@tool
def get_weather(date: str, localtion: str) -> dict | str:
    """
    Get weather
    
    Args:
        date (str): Date to get weather for
        location (str): Location to get weather for
    """


def get_openai_tools() -> List[dict]:
    functions = [
        get_balance,
        swap_token,
        get_weather,
    ]

    tools = [convert_to_openai_tool(f) for f in functions]
    return tools