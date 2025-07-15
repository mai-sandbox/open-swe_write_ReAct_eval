"""
Basic LangGraph AI Assistant with Search and Calculator Tools

This module implements a conversational AI assistant using LangGraph that can:
1. Have natural conversations with users
2. Search the web for current information using TavilySearch
3. Perform basic mathematical calculations using a custom calculator tool

The assistant uses Anthropic Claude as the LLM and maintains conversation memory.
"""

import os
from typing import Annotated, Any, Dict, List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv()


class State(TypedDict):
    """State schema for the LangGraph agent with message history."""
    messages: Annotated[List[BaseMessage], add_messages]


@tool
def calculator(expression: str) -> str:
    """
    Perform basic mathematical calculations.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        The result of the calculation as a string
        
    Raises:
        Exception: If the expression is invalid or cannot be evaluated
    """
    try:
        # Use eval with restricted globals for basic arithmetic operations
        allowed_names = {
            k: v for k, v in __builtins__.items()
            if k in ("abs", "round", "min", "max", "sum", "pow")
        }
        allowed_names.update({
            "__builtins__": {},
            "pow": pow,
            "abs": abs,
            "round": round,
        })
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# Initialize the LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# Initialize tools
search_tool = TavilySearch(max_results=2)
calc_tool = calculator
tools = [search_tool, calc_tool]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> Dict[str, List[BaseMessage]]:
    """
    Main chatbot node that processes messages and invokes the LLM with tools.
    
    Args:
        state: Current state containing message history
        
    Returns:
        Updated state with new message from the LLM
    """
    try:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        # Handle LLM invocation errors gracefully
        from langchain_core.messages import AIMessage
        error_message = AIMessage(
            content=f"I apologize, but I encountered an error: {str(e)}. Please try again."
        )
        return {"messages": [error_message]}


# Create the StateGraph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

# Add memory for conversation persistence
memory = MemorySaver()

# Compile the graph with checkpointer for memory
graph = graph_builder.compile(checkpointer=memory)

# Export the compiled graph as required by the evaluation script
compiled_graph = graph

