"""
LangGraph-based chat assistant with search and calculator tools.

This module implements a conversational AI agent that can:
- Have natural conversations with users
- Search the web for current information using TavilySearch
- Perform mathematical calculations using a custom calculator tool
- Route between conversation and tool usage intelligently
"""

from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    """State schema for the chat assistant graph."""
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
        # Use eval with restricted globals for basic math operations
        allowed_names = {
            k: v for k, v in __builtins__.items() 
            if k in ('abs', 'round', 'min', 'max', 'sum', 'pow')
        }
        allowed_names.update({
            '__builtins__': {},
            'pow': pow,
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
        })
        
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# Initialize the search tool
search_tool = TavilySearch(max_results=2)

# Define the tools list
tools = [search_tool, calculator]

# Initialize the LLM with Anthropic Claude
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> Dict[str, List[BaseMessage]]:
    """
    Main chatbot node that processes messages and decides whether to use tools.
    
    Args:
        state: Current state containing message history
        
    Returns:
        Updated state with new message from the assistant
    """
    try:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        # Graceful error handling for LLM failures
        from langchain_core.messages import AIMessage
        error_message = AIMessage(
            content=f"I apologize, but I encountered an error while processing your request: {str(e)}"
        )
        return {"messages": [error_message]}


# Create the StateGraph
graph_builder = StateGraph(State)

# Add nodes to the graph
graph_builder.add_node("chatbot", chatbot)

# Create tool node for handling tool execution
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add edges to the graph
graph_builder.add_edge(START, "chatbot")

# Add conditional edges from chatbot
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Add edge from tools back to chatbot
graph_builder.add_edge("tools", "chatbot")

# Compile the graph
graph = graph_builder.compile()

# Export the compiled graph (required for evaluation)
compiled_graph = graph

