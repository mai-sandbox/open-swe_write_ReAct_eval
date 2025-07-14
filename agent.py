"""
Basic chat assistant with LangGraph, search and calculator tools.

This module implements a conversational agent that can:
- Have normal conversations with users
- Search for information online using Tavily
- Perform basic mathematical calculations
"""

import os
from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv()


class State(TypedDict):
    """State schema for the chat assistant."""
    messages: Annotated[List[Any], add_messages]


@tool
def calculator(expression: str) -> str:
    """
    Perform basic arithmetic calculations.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        The result of the calculation as a string
    """
    try:
        # Only allow basic arithmetic operations for security
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic arithmetic operations (+, -, *, /, parentheses) and numbers are allowed."
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


# Initialize tools
search_tool = TavilySearch(max_results=2)
tools = [search_tool, calculator]

# Initialize LLM with tools
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> Dict[str, List[Any]]:
    """Main chatbot node that processes messages and decides whether to use tools."""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build the graph
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

# Export the compiled graph as required
compiled_graph = graph

