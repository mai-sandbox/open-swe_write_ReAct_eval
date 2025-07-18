"""
Basic LangGraph AI Assistant with Search and Calculator Tools

This module implements a conversational AI assistant that can:
- Have natural conversations with users
- Search the web for current information
- Perform mathematical calculations
- Route between tools and conversation appropriately
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
    """State schema for the chatbot with message history."""
    messages: Annotated[List[Any], add_messages]


# Initialize the LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


# Define the web search tool
search_tool = TavilySearch(max_results=2)


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
        # Only allow basic mathematical operations for security
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic mathematical operations are allowed (+, -, *, /, parentheses, and numbers)"
        
        result = eval(expression)
        return f"The result is: {result}"
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed"
    except Exception as e:
        return f"Error: Could not evaluate the expression. {str(e)}"


# Create tools list
tools = [search_tool, calculator]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> Dict[str, List[Any]]:
    """Main chatbot node that processes messages and decides on tool usage."""
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

# Export the compiled graph (required for evaluation)
compiled_graph = graph

