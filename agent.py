"""
LangGraph-based chat assistant with search and calculator tools.

This module implements a conversational agent that can:
- Have natural conversations with users
- Search the web for current information using Tavily
- Perform basic mathematical calculations
- Route between tools and conversation appropriately
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
    """State schema for the chat assistant."""
    messages: Annotated[List[BaseMessage], add_messages]


@tool
def calculator(expression: str) -> str:
    """
    Perform basic mathematical calculations.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        The result of the calculation as a string
    """
    try:
        # Use eval with restricted globals for basic arithmetic
        # Only allow basic math operations and built-in functions
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
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# Initialize the LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# Initialize tools
search_tool = TavilySearch(max_results=3)
calc_tool = calculator

tools = [search_tool, calc_tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> Dict[str, Any]:
    """Main chatbot node that processes messages and decides on tool usage."""
    try:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        # Graceful error handling for LLM failures
        from langchain_core.messages import AIMessage
        error_msg = AIMessage(content=f"I encountered an error: {str(e)}. Please try again.")
        return {"messages": [error_msg]}


# Build the graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools)
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

