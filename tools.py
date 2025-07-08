"""
Tools module for LangGraph chatbot implementation.

This module contains tool implementations for web search and mathematical calculations.
All tools include comprehensive type hints and error handling as required by project standards.
"""

import ast
import operator
import re
from typing import Any, Dict, List, Union

from langchain_core.tools import tool
from langchain_tavily import TavilySearch


# Initialize TavilySearch tool with max_results=2 as per LangGraph documentation pattern
tavily_search = TavilySearch(max_results=2)


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information using Tavily Search Engine.
    
    This tool is useful for finding up-to-date information, current events,
    facts, news, and any information that might not be in the LLM's training data.
    
    Args:
        query: The search query string to look up information for
        
    Returns:
        A string containing the search results with relevant information
        
    Raises:
        Exception: If the search fails due to API issues or network problems
    """
    try:
        # Use the TavilySearch tool to perform the search
        results = tavily_search.invoke({"query": query})
        
        # Handle case where results might be empty or None
        if not results:
            return f"No search results found for query: {query}"
            
        return results
        
    except Exception as e:
        # Graceful error handling for API failures, network issues, etc.
        return f"Search failed for query '{query}': {str(e)}"


@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations safely using Python's eval() with input validation.
    
    This tool can handle basic arithmetic operations (+, -, *, /, **, %), 
    mathematical functions, and expressions with parentheses.
    
    Args:
        expression: A mathematical expression string to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        A string containing the calculation result or an error message
        
    Raises:
        Exception: If the expression is invalid or contains unsafe operations
    """
    try:
        # Input validation: only allow safe mathematical expressions
        # Remove whitespace and check for basic safety
        cleaned_expr = expression.strip()
        
        # Basic validation: only allow numbers, operators, parentheses, and dots
        if not re.match(r'^[0-9+\-*/().%\s**]+$', cleaned_expr):
            return f"Invalid expression: '{expression}'. Only basic mathematical operations are allowed."
        
        # Use ast.literal_eval for safer evaluation when possible, fallback to eval for complex expressions
        try:
            # For simple numeric literals, use ast.literal_eval
            result = ast.literal_eval(cleaned_expr)
        except (ValueError, SyntaxError):
            # For mathematical expressions, use eval with restricted globals
            result = eval(cleaned_expr, {"__builtins__": {}}, {})
        
        return f"{expression} = {result}"
        
    except Exception as e:
        # Comprehensive error handling for invalid expressions, division by zero, etc.
        return f"Calculation error for '{expression}': {str(e)}"


# Export the tools list for use in the main agent
tools: List[Any] = [web_search, calculator]

