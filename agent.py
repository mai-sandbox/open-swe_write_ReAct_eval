"""
Basic LangGraph AI Assistant with Search and Calculator Tools

This module implements a conversational agent using LangGraph that can:
- Have natural conversations with users
- Search the web for current information using TavilySearch
- Perform basic mathematical calculations using a custom calculator tool
- Route intelligently between conversation and tool usage
"""

from typing import Annotated, Any, Dict, List, Union
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    """
    State schema for the LangGraph agent.
    
    The messages field uses add_messages to properly handle conversation history
    by appending new messages while maintaining proper message ordering.
    """
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
        Exception: If the expression is invalid or cannot be evaluated safely
    """
    try:
        # Use eval with restricted globals for basic arithmetic safety
        allowed_names = {
            k: v for k, v in __builtins__.items() 
            if k in ("abs", "round", "min", "max", "sum", "pow")
        }
        allowed_names.update({"__builtins__": {}})
        
        result = eval(expression, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# Initialize the search tool with limited results for efficiency
search_tool = TavilySearch(max_results=2)

# Create the tools list
tools = [search_tool, calculator]

# Initialize the LLM with Anthropic Claude and bind tools for tool calling
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> Dict[str, List[BaseMessage]]:
    """
    Main chatbot node that processes user messages and generates responses.
    
    This function invokes the LLM with the current conversation history.
    The LLM will decide whether to respond directly or call tools based on the user's input.
    
    Args:
        state: Current state containing conversation messages
        
    Returns:
        Dictionary with updated messages including the LLM's response
    """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Build the StateGraph
graph_builder = StateGraph(State)

# Add nodes to the graph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Define the graph structure with proper routing
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# After tools are executed, return to chatbot for response generation
graph_builder.add_edge("tools", "chatbot")

# Compile the graph
graph = graph_builder.compile()

# Export the compiled graph for evaluation
compiled_graph = graph

