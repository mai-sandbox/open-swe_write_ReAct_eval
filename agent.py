"""
Basic chat assistant with LangGraph, search and calculator tools.

This module implements a conversational agent that can:
- Have normal conversations with users
- Search for information online using Tavily
- Perform basic mathematical calculations
"""

import os
import logging
from typing import Annotated, Dict, List, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class State(TypedDict):
    """State schema for the chat assistant."""
    messages: Annotated[List[BaseMessage], add_messages]


def validate_state(state: State) -> bool:
    """Validate the state structure and content."""
    try:
        return isinstance(state, dict) and "messages" in state and isinstance(state["messages"], list)
    except Exception as e:
        logger.error(f"State validation error: {e}")
        return False


@tool
def calculator(expression: str) -> str:
    """
    Perform basic arithmetic calculations.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        The result of the calculation as a string
    """
    if not isinstance(expression, str):
        return "Error: Expression must be a string"
    
    if not expression.strip():
        return "Error: Expression cannot be empty"
    
    try:
        # Only allow basic arithmetic operations for security
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic arithmetic operations (+, -, *, /, parentheses) and numbers are allowed."
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except ZeroDivisionError:
        return f"Error: Division by zero in expression {expression}"
    except SyntaxError:
        return f"Error: Invalid syntax in expression {expression}"
    except ValueError as e:
        return f"Error: Invalid value in expression {expression}: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in calculator: {e}")
        return f"Error calculating {expression}: {str(e)}"


def initialize_search_tool() -> Optional[TavilySearch]:
    """Initialize the Tavily search tool with error handling."""
    try:
        # Check for required API key
        if not os.getenv("TAVILY_API_KEY"):
            logger.warning("TAVILY_API_KEY not found in environment variables")
            return None
        
        search_tool = TavilySearch(max_results=2)
        logger.info("Tavily search tool initialized successfully")
        return search_tool
    except Exception as e:
        logger.error(f"Failed to initialize Tavily search tool: {e}")
        return None


def initialize_llm() -> Optional[BaseChatModel]:
    """Initialize the LLM with error handling."""
    try:
        # Check for required API key
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.error("ANTHROPIC_API_KEY not found in environment variables")
            raise ValueError("ANTHROPIC_API_KEY is required but not found in environment")
        
        llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise


def create_tools_list() -> List[BaseTool]:
    """Create and validate the tools list."""
    tools: List[BaseTool] = []
    
    # Always add calculator tool as it doesn't require external API
    tools.append(calculator)
    
    # Try to add search tool if available
    search_tool = initialize_search_tool()
    if search_tool is not None:
        tools.append(search_tool)
    else:
        logger.warning("Search tool not available - continuing without search functionality")
    
    return tools


def chatbot(state: State) -> Dict[str, List[BaseMessage]]:
    """
    Main chatbot node that processes messages and decides whether to use tools.
    
    Args:
        state: The current state containing messages
        
    Returns:
        Updated state with new messages
        
    Raises:
        ValueError: If state is invalid
        RuntimeError: If LLM invocation fails
    """
    try:
        # Validate input state
        if not validate_state(state):
            raise ValueError("Invalid state structure")
        
        if not state["messages"]:
            logger.warning("No messages in state")
            return {"messages": [AIMessage(content="Hello! How can I help you today?")]}
        
        # Invoke LLM with error handling
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
        
    except Exception as e:
        logger.error(f"Error in chatbot node: {e}")
        error_message = AIMessage(
            content="I apologize, but I encountered an error processing your request. Please try again."
        )
        return {"messages": [error_message]}


# Initialize components with error handling
try:
    tools = create_tools_list()
    llm = initialize_llm()
    llm_with_tools = llm.bind_tools(tools)
    logger.info(f"Initialized {len(tools)} tools successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    raise


# Build the graph with error handling
try:
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    # Add edges
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    
    # Compile the graph
    graph = graph_builder.compile()
    logger.info("Graph compiled successfully")
    
except Exception as e:
    logger.error(f"Failed to build graph: {e}")
    raise

# Export the compiled graph as required
compiled_graph = graph






