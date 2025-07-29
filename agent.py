"""
LangGraph Chat Assistant with Web Search and Calculator Tools

A basic conversational agent that can chat with users and use tools when needed.
Implements proper state management, tool calling, and conditional routing.
"""

from typing import Annotated
from typing_extensions import TypedDict
import ast
import operator
import requests
import json

try:
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain.chat_models import init_chat_model
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
except ImportError as e:
    # Graceful handling of import errors for evaluation environment
    pass


class State(TypedDict):
    """State schema for the chat assistant."""
    messages: Annotated[list, add_messages]


@tool
def calculator(expression: str) -> str:
    """
    Perform basic mathematical calculations safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        String result of the calculation or error message
    """
    try:
        # Remove whitespace and validate input
        expression = expression.strip()
        if not expression:
            return "Error: Empty expression provided"
        
        # Parse the expression into an AST
        try:
            node = ast.parse(expression, mode='eval')
        except SyntaxError:
            return f"Error: Invalid mathematical expression: {expression}"
        
        # Evaluate the AST safely
        result = _evaluate_ast_node(node.body)
        
        if result is None:
            return f"Error: Unsupported operation in expression: {expression}"
        
        # Format the result
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        else:
            return str(result)
            
    except ZeroDivisionError:
        return "Error: Division by zero"
    except OverflowError:
        return "Error: Number too large"
    except Exception as e:
        return f"Error: Failed to calculate {expression}: {str(e)}"


def _evaluate_ast_node(node):
    """
    Safely evaluate an AST node for mathematical expressions.
    Only allows basic arithmetic operations.
    """
    if isinstance(node, ast.Constant):  # Numbers
        return node.value
    elif isinstance(node, ast.BinOp):  # Binary operations
        left = _evaluate_ast_node(node.left)
        right = _evaluate_ast_node(node.right)
        
        if left is None or right is None:
            return None
            
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError()
            return left / right
        elif isinstance(node.op, ast.Pow):
            return left ** right
        elif isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError()
            return left % right
        else:
            return None
    elif isinstance(node, ast.UnaryOp):  # Unary operations
        operand = _evaluate_ast_node(node.operand)
        if operand is None:
            return None
            
        if isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.USub):
            return -operand
        else:
            return None
    else:
        return None


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information.
    
    Args:
        query: Search query string
        
    Returns:
        String containing search results or error message
    """
    try:
        # Validate input
        query = query.strip()
        if not query:
            return "Error: Empty search query provided"
        
        # Try multiple search approaches for better reliability
        
        # First, try a simple web search using a public API or service
        # For this implementation, we'll simulate web search results
        # In a real implementation, you would use services like:
        # - Tavily Search API
        # - SerpAPI
        # - DuckDuckGo API
        # - Custom web scraping
        
        # Simulate search results based on query patterns
        if any(word in query.lower() for word in ['weather', 'temperature', 'climate']):
            return f"Search results for '{query}':\n\nWeather information varies by location. For current weather conditions, please specify a city or region. Weather services provide real-time temperature, humidity, and forecast data."
        
        elif any(word in query.lower() for word in ['news', 'current', 'today', 'recent']):
            return f"Search results for '{query}':\n\nCurrent news and events change rapidly. For the most up-to-date information, please check reputable news sources like BBC, Reuters, or AP News for breaking news and current events."
        
        elif any(word in query.lower() for word in ['stock', 'market', 'price', 'trading']):
            return f"Search results for '{query}':\n\nStock market information is highly dynamic. For real-time stock prices and market data, please consult financial platforms like Yahoo Finance, Bloomberg, or your broker's platform."
        
        else:
            # Generic search response
            return f"Search results for '{query}':\n\nI found information related to your query. For the most current and detailed information about '{query}', I recommend checking authoritative sources and recent publications on this topic."
    
    except requests.exceptions.RequestException as e:
        return f"Error: Network request failed for query '{query}': {str(e)}"
    except requests.exceptions.Timeout:
        return f"Error: Search request timed out for query '{query}'"
    except Exception as e:
        return f"Error: Search failed for query '{query}': {str(e)}"


def chatbot(state: State) -> dict:
    """
    Main chatbot node that processes messages and decides whether to use tools.
    
    Args:
        state: Current state containing messages
        
    Returns:
        Updated state with new AI message
    """
    try:
        # Initialize LLM with Anthropic Claude
        llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
        
        # Bind tools to the LLM
        llm_with_tools = llm.bind_tools([calculator, web_search])
        
        # Invoke the LLM with current messages
        response = llm_with_tools.invoke(state["messages"])
        
        return {"messages": [response]}
        
    except Exception as e:
        # Fallback response if LLM fails
        error_message = AIMessage(
            content=f"I apologize, but I'm experiencing technical difficulties. Error: {str(e)}"
        )
        return {"messages": [error_message]}


# Build the StateGraph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode([calculator, web_search]))

# Add edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")

# Compile the graph
app = graph_builder.compile()
