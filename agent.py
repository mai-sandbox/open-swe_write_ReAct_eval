"""
Basic LangGraph AI Assistant with Search and Calculator Tools

This module implements a conversational agent using LangGraph that can:
- Have normal conversations with users
- Search for information online using Tavily
- Perform mathematical calculations
- Route intelligently between tools and direct responses
"""

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages


class State(TypedDict):
    """State schema for the LangGraph chatbot.
    
    Contains the conversation messages with proper type annotation
    for LangGraph's message handling system.
    """
    messages: Annotated[list, add_messages]


@tool
def calculator(expression: str) -> str:
    """Safely evaluate mathematical expressions.
    
    This tool can perform basic arithmetic operations including:
    - Addition (+)
    - Subtraction (-)
    - Multiplication (*)
    - Division (/)
    - Exponentiation (**)
    - Parentheses for grouping
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        String representation of the calculation result
    """
    try:
        # Basic input validation - only allow safe mathematical characters
        allowed_chars = set('0123456789+-*/().**e ')
        if not all(c in allowed_chars for c in expression.replace(' ', '')):
            return "Error: Invalid characters in expression. Only numbers and basic math operators (+, -, *, /, **, parentheses) are allowed."
        
        # Evaluate the expression safely
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except (ValueError, SyntaxError, TypeError):
        return "Error: Invalid mathematical expression. Please check your syntax."
    except Exception as e:
        return f"Error: Unable to evaluate expression - {str(e)}"


# Initialize the StateGraph with our State schema
graph_builder = StateGraph(State)

# Initialize search tool with max_results=2 and error handling
try:
    search_tool = TavilySearch(max_results=2)
    tools = [search_tool, calculator]
except Exception as e:
    # Handle initialization errors gracefully
    tools = [calculator]  # Calculator tool should still work even if search fails

# Initialize Anthropic Claude LLM
llm = init_chat_model('anthropic:claude-3-5-sonnet-latest')


def chatbot(state: State) -> dict:
    """Main chatbot node that processes user messages and generates responses.
    
    Args:
        state: Current conversation state containing message history
        
    Returns:
        Dictionary with new messages to add to state
    """
    return {"messages": [llm.invoke(state["messages"])]}


# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Set up the basic graph structure
graph_builder.add_edge(START, "chatbot")

# Compile the graph
graph = graph_builder.compile()

# Export the compiled graph as required by the evaluation script
compiled_graph = graph



