"""
Basic LangGraph AI Assistant with Search and Calculator Tools

This module implements a conversational AI assistant that can:
- Have natural conversations with users
- Search the web for current information using Tavily
- Perform mathematical calculations
- Route intelligently between conversation and tool usage
"""

from typing import Annotated
import os

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# Define the state schema for the conversation
class State(TypedDict):
    """State schema for the LangGraph assistant."""
    messages: Annotated[list[BaseMessage], add_messages]


# Initialize the LLM with Anthropic Claude
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


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
        # Use eval with restricted globals for basic math operations
        import builtins
        allowed_functions = ("abs", "round", "min", "max", "sum", "pow")
        allowed_names = {
            name: getattr(builtins, name) 
            for name in allowed_functions 
            if hasattr(builtins, name)
        }
        # Add math operators and constants
        allowed_names.update({"__builtins__": {}})
        
        result = eval(expression, allowed_names)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# Initialize tools with error handling for missing API keys
tools = [calculator]  # Always include calculator

# Try to initialize TavilySearch, but make it optional if API key is missing
try:
    if os.getenv("TAVILY_API_KEY"):
        search_tool = TavilySearch(max_results=2)
        tools.append(search_tool)
    else:
        # Create a dummy search tool that explains the API key is missing
        @tool
        def search_unavailable(query: str) -> str:
            """Search tool is unavailable - TAVILY_API_KEY not configured."""
            return "Search functionality is currently unavailable. Please configure the TAVILY_API_KEY environment variable to enable web search."
        
        tools.append(search_unavailable)
except Exception as e:
    # If TavilySearch fails to initialize for any reason, provide a fallback
    @tool
    def search_error(query: str) -> str:
        """Search tool encountered an error during initialization."""
        return f"Search functionality encountered an error during initialization: {str(e)}. Please check your TAVILY_API_KEY configuration."
    
    tools.append(search_error)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> dict[str, list[BaseMessage]]:
    """Main chatbot node that processes messages and decides on tool usage."""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Create the state graph
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
# After tool execution, return to chatbot for next decision
graph_builder.add_edge("tools", "chatbot")

# Compile the graph
compiled_graph = graph_builder.compile()




