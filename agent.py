"""
Basic chat assistant with LangGraph, search and calculator tools.

This module implements a conversational agent that can:
- Have normal conversations with users
- Search for information online using Tavily
- Perform basic mathematical calculations
"""

from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    """State schema for the chat assistant."""
    messages: Annotated[List[Any], add_messages]


# Initialize the LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


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
        # Use eval safely for basic arithmetic operations
        # In production, consider using a more secure math parser
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# Initialize tools
search_tool = TavilySearch(max_results=2)
tools = [search_tool, calculator]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> Dict[str, List[Any]]:
    """
    Main chatbot node that processes messages and decides whether to use tools.
    
    Args:
        state: Current state containing conversation messages
        
    Returns:
        Updated state with new message from the LLM
    """
    try:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        # Handle LLM errors gracefully without logging
        error_msg = f"I encountered an error processing your request: {str(e)}"
        return {"messages": [{"role": "assistant", "content": error_msg}]}


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
compiled_graph = graph_builder.compile()

