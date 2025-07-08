"""
Main chatbot agent implementation using LangGraph StateGraph.

This module implements a LangGraph-based chatbot that can handle conversations,
web search, and mathematical calculations using the tools defined in tools.py.
"""

import os
from typing import Annotated, TypedDict
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import tools

# Load environment variables
load_dotenv()


class State(TypedDict):
    """
    State TypedDict for the chatbot agent using add_messages for message handling.
    
    This defines the state structure that will be passed between nodes in the graph.
    The messages field uses add_messages to properly handle message accumulation.
    """
    messages: Annotated[list[BaseMessage], add_messages]


def init_chat_model() -> Any:
    """
    Initialize the Anthropic Claude chat model with tools bound.
    
    Returns:
        ChatAnthropic: Configured chat model with tools bound for tool calling
        
    Raises:
        ValueError: If ANTHROPIC_API_KEY environment variable is not set
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    # Initialize the chat model
    llm = ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022",
        timeout=60,
        stop=None,
        # API key will be read from ANTHROPIC_API_KEY environment variable
    )
    
    # Bind tools to the model for tool calling capability
    return llm.bind_tools(tools)


def chatbot_node(state: State) -> dict[str, list[BaseMessage]]:
    """
    Chatbot node function that processes messages and generates responses.
    
    This node handles the main conversation logic and determines whether to
    respond directly or call tools based on the user's input.
    
    Args:
        state: Current state containing the conversation messages
        
    Returns:
        Dictionary with updated messages including the model's response
    """
    # Get the chat model with tools bound
    llm_with_tools = init_chat_model()
    
    # Generate response from the model
    response = llm_with_tools.invoke(state["messages"])
    
    # Return the updated state with the new message
    return {"messages": [response]}


# Create the StateGraph with the State schema
graph = StateGraph(State)

# Add nodes to the graph
graph.add_node("chatbot", chatbot_node)
graph.add_node("tools", ToolNode(tools))

# Add edges to the graph
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", tools_condition)
graph.add_edge("tools", "chatbot")

# Compile the graph
app = graph.compile()






