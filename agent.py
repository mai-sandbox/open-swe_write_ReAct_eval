"""
LangGraph Chat Assistant with Web Search and Calculator Tools

A basic conversational agent that can:
- Have natural conversations
- Search the web for current information
- Perform basic mathematical calculations
- Route intelligently between tools and conversation
"""

from typing import Annotated, Dict, Any
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    """State schema for the chat assistant."""
    messages: Annotated[list[BaseMessage], add_messages]


# Initialize the LLM
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# Define web search tool
search_tool = TavilySearch(max_results=2)

# Define calculator tools
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first number."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide the first number by the second number."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# Collect all tools
tools = [search_tool, add, subtract, multiply, divide]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> Dict[str, Any]:
    """
    Main chatbot node that processes user messages and decides whether to use tools.
    
    Args:
        state: Current conversation state containing messages
        
    Returns:
        Updated state with the LLM's response
    """
    try:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        # Handle LLM errors gracefully
        from langchain_core.messages import AIMessage
        error_response = AIMessage(
            content=f"I apologize, but I encountered an error while processing your request. Please try again."
        )
        return {"messages": [error_response]}


# Create the graph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

# Add edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")

# Compile the graph
app = graph_builder.compile()
