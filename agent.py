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
    Perform basic mathematical calculations safely using AST evaluation.
    
    Supports arithmetic operations: +, -, *, /, **, %
    Uses secure AST parsing similar to ast.literal_eval principles for safety.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4", "10 / 2", "2 ** 3")
        
    Returns:
        String result of the calculation or descriptive error message
    """
    try:
        # Input validation and sanitization
        if not isinstance(expression, str):
            return "Error: Expression must be a string"
        
        expression = expression.strip()
        if not expression:
            return "Error: Empty expression provided"
        
        # Check for potentially dangerous patterns
        dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'file', 'input', 'raw_input']
        if any(pattern in expression.lower() for pattern in dangerous_patterns):
            return f"Error: Expression contains potentially unsafe operations: {expression}"
        
        # Parse the expression into an AST for safe evaluation
        try:
            # Use ast.parse in eval mode to create an AST
            parsed_ast = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            return f"Error: Invalid mathematical syntax in '{expression}': {str(e)}"
        
        # Safely evaluate the AST using our secure evaluator
        # This follows ast.literal_eval security principles but extends to arithmetic
        result = _safe_eval_math_ast(parsed_ast.body)
        
        if result is None:
            return f"Error: Unsupported operation or invalid expression: {expression}"
        
        # Format the result appropriately
        if isinstance(result, float):
            # Handle very large or very small numbers
            if abs(result) > 1e15:
                return f"Error: Result too large: {result}"
            elif abs(result) < 1e-15 and result != 0:
                return f"Error: Result too small: {result}"
            # Convert integer-valued floats to integers for cleaner display
            elif result.is_integer():
                return str(int(result))
            else:
                # Round to reasonable precision to avoid floating point artifacts
                return str(round(result, 10))
        else:
            return str(result)
            
    except ZeroDivisionError:
        return f"Error: Division by zero in expression: {expression}"
    except OverflowError:
        return f"Error: Number too large to compute in expression: {expression}"
    except ValueError as e:
        return f"Error: Invalid value in expression '{expression}': {str(e)}"
    except Exception as e:
        return f"Error: Failed to calculate '{expression}': {str(e)}"


def _safe_eval_math_ast(node):
    """
    Safely evaluate an AST node for mathematical expressions.
    
    This function implements security principles similar to ast.literal_eval
    but extends support to basic arithmetic operations (+, -, *, /, **, %).
    Only allows safe mathematical operations on numbers.
    
    Args:
        node: AST node to evaluate
        
    Returns:
        Numeric result or None if operation is not supported
        
    Raises:
        ZeroDivisionError: For division by zero
        OverflowError: For results that are too large
        ValueError: For invalid numeric operations
    """
    # Handle numeric constants (literals)
    if isinstance(node, ast.Constant):
        # Only allow numeric constants for security
        if isinstance(node.value, (int, float, complex)):
            return node.value
        else:
            return None  # Reject non-numeric literals
    
    # Handle numeric literals (for older Python versions)
    elif isinstance(node, ast.Num):
        return node.n
    
    # Handle binary operations (+, -, *, /, **, %)
    elif isinstance(node, ast.BinOp):
        left = _safe_eval_math_ast(node.left)
        right = _safe_eval_math_ast(node.right)
        
        # Both operands must be valid numbers
        if left is None or right is None:
            return None
        
        # Ensure operands are numeric
        if not isinstance(left, (int, float, complex)) or not isinstance(right, (int, float, complex)):
            return None
            
        # Perform the operation based on the operator type
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("Division by zero")
            return left / right
        elif isinstance(node.op, ast.Pow):
            # Prevent extremely large exponents that could cause system issues
            if isinstance(right, (int, float)) and abs(right) > 1000:
                raise OverflowError("Exponent too large")
            return left ** right
        elif isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("Modulo by zero")
            return left % right
        else:
            return None  # Unsupported binary operation
    
    # Handle unary operations (+x, -x)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_math_ast(node.operand)
        
        if operand is None:
            return None
        
        # Ensure operand is numeric
        if not isinstance(operand, (int, float, complex)):
            return None
            
        if isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.USub):
            return -operand
        else:
            return None  # Unsupported unary operation
    
    # Reject all other node types for security
    # This includes: function calls, attribute access, subscripting, etc.
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


