import re
import inspect
import requests
import pandas as pd
import yfinance as yf
import concurrent.futures
import io
import base64
from datetime import date

from typing import List
from bs4 import BeautifulSoup
from utils import inference_logger
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from lxml import etree

@tool
def code_interpreter(code_markdown: str) -> dict | str:
    """
    Execute the provided Python code string on the terminal using exec.

    The string should contain valid, executable and pure Python code in markdown syntax.
    Code should also import any required Python packages.

    Args:
        code_markdown (str): The Python code with markdown syntax to be executed.
            For example: ```python\n<code-string>\n```

    Returns:
        dict | str: A dictionary containing variables declared and values returned by function calls,
            or an error message if an exception occurred.

    Note:
        Use this function with caution, as executing arbitrary code can pose security risks.
    """
    try:
        # Extracting code from Markdown code block
        code_lines = code_markdown.strip().split('\n')
        if code_lines[0].strip().startswith("```"):
             code_lines = code_lines[1:]
        if code_lines[-1].strip() == "```":
             code_lines = code_lines[:-1]
        code_without_markdown = '\n'.join(code_lines)

        # Create a new namespace for code execution
        exec_namespace = {}
        
        # Capture stdout
        stdout_capture = io.StringIO()
        
        # Execute the code in the new namespace, redirecting stdout
        exec(code_without_markdown, exec_namespace)
        
        # After execution, check for plots
        result_dict = {}
        if 'plt' in exec_namespace and hasattr(exec_namespace['plt'], 'get_fignums') and exec_namespace['plt'].get_fignums():
            buf = io.BytesIO()
            exec_namespace['plt'].savefig(buf, format='png')
            buf.seek(0)
            # Encode the plot to base64
            result_dict['plot_image'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            exec_namespace['plt'].close() # Close the plot to free memory
        
        # Restore stdout and get the captured output
        output = stdout_capture.getvalue()
        if output:
            result_dict['stdout'] = output

        return result_dict

    except Exception as e:
        error_message = f"An error occurred: {e}"
        inference_logger.error(error_message)
        return error_message

@tool
def google_search_and_scrape(query: str) -> dict:
    """
    Performs a Google search for the given query, retrieves the top search result URLs,
    and scrapes the text content and table data from those pages in parallel.

    Args:
        query (str): The search query.
    Returns:
        list: A list of dictionaries containing the URL, text content, and table data for each scraped page.
    """
    num_results = 2
    url = 'https://www.google.com/search'
    params = {'q': query, 'num': num_results}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.3'}
    
    inference_logger.info(f"Performing google search with query: {query}\nplease wait...")
    response = requests.get(url, params=params, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Use lxml for parsing and robust XPath selectors
    dom = etree.HTML(str(soup))
    
    # This XPath is more robust as it doesn't rely on volatile class names.
    # It finds the main results block and then iterates through the individual result containers.
    search_results = dom.xpath('//div[h3 and .//a]')
    
    urls = [result.xpath('.//a/@href')[0] for result in search_results if result.xpath('.//a/@href')]

    # If no URLs are found, it's a failure. Return a clear error.
    if not urls:
        return {"error": "The Google Search tool failed to retrieve any results. The page format may have changed, or the search returned no results."}
    
    inference_logger.info(f"Scraping text from {len(urls)} urls, please wait...")
    [inference_logger.info(url) for url in urls]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(lambda url: (url, requests.get(url, headers=headers).text if isinstance(url, str) else None), url) for url in urls[:num_results] if isinstance(url, str)]
        results = []
        for future in concurrent.futures.as_completed(futures):
            url, html = future.result()
            soup = BeautifulSoup(html, 'html.parser')
            paragraphs = [p.text.strip() for p in soup.find_all('p') if p.text.strip()]
            text_content = ' '.join(paragraphs)
            text_content = re.sub(r'\s+', ' ', text_content)
            table_data = [[cell.get_text(strip=True) for cell in row.find_all('td')] for table in soup.find_all('table') for row in table.find_all('tr')]
            if text_content or table_data:
                results.append({'url': url, 'content': text_content, 'tables': table_data})
    return results

@tool
def get_current_stock_price(symbol: str) -> float:
  """
  Get the current stock price for a given symbol.

  Args:
    symbol (str): The stock symbol.

  Returns:
    float: The current stock price, or None if an error occurs.
  """
  try:
    stock = yf.Ticker(symbol)
    # Use "regularMarketPrice" for regular market hours, or "currentPrice" for pre/post market
    current_price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
    return current_price if current_price else None
  except Exception as e:
    print(f"Error fetching current price for {symbol}: {e}")
    return None

@tool
def get_stock_fundamentals(symbol: str) -> dict:
    """
    Get fundamental data for a given stock symbol using yfinance API.

    Args:
        symbol (str): The stock symbol.

    Returns:
        dict: A dictionary containing fundamental data, or an error dictionary if data cannot be retrieved.
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        # yfinance can return a mostly empty dict if the ticker is invalid or data is missing.
        # We check for a key that should almost always be present for a valid public company.
        if not info or not info.get('longName'):
            return {"error": f"Could not retrieve fundamental data for symbol: {symbol}. The symbol may be invalid or no data is available."}

        fundamentals = {
            'symbol': symbol,
            'company_name': info.get('longName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('forwardPE'),
            'pb_ratio': info.get('priceToBook'),
            'dividend_yield': info.get('dividendYield'),
            'eps': info.get('trailingEps'),
            'beta': info.get('beta'),
            '52_week_high': info.get('fiftyTwoWeekHigh'),
            '52_week_low': info.get('fiftyTwoWeekLow')
        }
        # Clean up None values for a cleaner output to the model and UI
        return {k: v for k, v in fundamentals.items() if v is not None}

    except Exception as e:
        inference_logger.error(f"Error getting fundamentals for {symbol}: {e}")
        return {"error": f"An unexpected error occurred while fetching fundamentals: {e}"}

@tool
def get_financial_statements(symbol: str) -> dict:
    """
    Get financial statements for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    dict: Dictionary containing financial statements (income statement, balance sheet, cash flow statement).
    """
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        return financials
    except Exception as e:
        print(f"Error fetching financial statements for {symbol}: {e}")
        return {}

@tool
def get_key_financial_ratios(symbol: str) -> dict:
    """
    Get key financial ratios for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    dict: Dictionary containing key financial ratios.
    """
    try:
        stock = yf.Ticker(symbol)
        key_ratios = stock.info
        return key_ratios
    except Exception as e:
        print(f"Error fetching key financial ratios for {symbol}: {e}")
        return {}

@tool
def get_analyst_recommendations(symbol: str) -> pd.DataFrame:
    """
    Get analyst recommendations for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    pd.DataFrame: DataFrame containing analyst recommendations.
    """
    try:
        stock = yf.Ticker(symbol)
        recommendations = stock.recommendations
        return recommendations
    except Exception as e:
        print(f"Error fetching analyst recommendations for {symbol}: {e}")
        return pd.DataFrame()

@tool
def get_dividend_data(symbol: str) -> pd.DataFrame:
    """
    Get dividend data for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    pd.DataFrame: DataFrame containing dividend data.
    """
    try:
        stock = yf.Ticker(symbol)
        dividends = stock.dividends
        return dividends
    except Exception as e:
        print(f"Error fetching dividend data for {symbol}: {e}")
        return pd.DataFrame()

@tool
def get_todays_weather(location: str) -> dict:
    """
    Get the current weather for a given location.

    Args:
        location (str): The city or area for which to get the weather. E.g., "San Francisco", "London".

    Returns:
        dict: A dictionary containing the current weather conditions.
    """
    try:
        # wttr.in provides a simple JSON API
        url = f"https://wttr.in/{location}?format=j1"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        weather_data = response.json()
        
        # Extract the most relevant current information
        current_condition = weather_data.get('current_condition', [{}])[0]
        
        return {
            "location": weather_data.get('nearest_area', [{}])[0].get('value', location),
            "current_temp_c": current_condition.get('temp_C'),
            "current_temp_f": current_condition.get('temp_F'),
            "description": current_condition.get('weatherDesc', [{}])[0].get('value'),
            "wind_speed_mph": current_condition.get('windspeedMiles'),
            "humidity": current_condition.get('humidity'),
            "precipitation_mm": current_condition.get('precipMM')
        }
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            return {"error": f"Location '{location}' not found."}
        else:
            return {"error": f"HTTP error occurred: {http_err}"}
    except Exception as e:
        inference_logger.error(f"Error fetching weather for {location}: {e}")
        return {"error": f"An error occurred while fetching weather data: {e}"}

@tool
def get_todays_date() -> str:
    """
    Returns today's date in YYYY-MM-DD format.
    This function takes no arguments.
    """
    return date.today().isoformat()

@tool
def get_company_news(symbol: str) -> pd.DataFrame:
    """
    Get company news and press releases for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    pd.DataFrame: DataFrame containing company news and press releases.
    """
    try:
        news = yf.Ticker(symbol).news
        return news
    except Exception as e:
        print(f"Error fetching company news for {symbol}: {e}")
        return pd.DataFrame()

@tool
def get_technical_indicators(symbol: str) -> pd.DataFrame:
    """
    Get technical indicators for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    pd.DataFrame: DataFrame containing technical indicators.
    """
    try:
        indicators = yf.Ticker(symbol).history(period="max")
        return indicators
    except Exception as e:
        print(f"Error fetching technical indicators for {symbol}: {e}")
        return pd.DataFrame()

@tool
def get_company_profile(symbol: str) -> dict:
    """
    Get company profile and overview for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    dict: Dictionary containing company profile and overview.
    """
    try:
        profile = yf.Ticker(symbol).info
        return profile
    except Exception as e:
        print(f"Error fetching company profile for {symbol}: {e}")
        return {}

def get_openai_tools() -> List[dict]:
    all_funcs = [
        code_interpreter,
        google_search_and_scrape,
        get_current_stock_price,
        get_company_news,
        get_company_profile,
        get_stock_fundamentals,
        get_financial_statements,
        get_key_financial_ratios,
        get_analyst_recommendations,
        get_dividend_data,
        get_todays_weather,
        get_todays_date,
        get_technical_indicators
    ]
    tools = [convert_to_openai_tool(f) for f in all_funcs]
    return tools