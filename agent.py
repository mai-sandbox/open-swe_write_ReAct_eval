"""
LangGraph chat assistant with search and calculator tools.

This module implements a conversational agent using LangGraph that can:
- Have normal conversations with users
- Search for information online using TavilySearch
- Perform basic mathematical calculations
"""

from typing import Annotated, Any, Dict, List, Union
import operator
import re

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from typing_extensions import TypedDict

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
    """
    try:
        # Sanitize the expression to only allow safe mathematical operations
        # Remove any non-mathematical characters for security
        safe_expression = re.sub(r'[^0-9+\-*/().\s]', '', expression)
        
        if not safe_expression.strip():
            return "Error: Invalid mathematical expression"
        
        # Evaluate the expression safely
        result = eval(safe_expression)
        return f"The result of {expression} is {result}"
    
    except ZeroDivisionError:
        return "Error: Division by zero"
    except (SyntaxError, ValueError, TypeError) as e:
        return f"Error: Invalid mathematical expression - {str(e)}"
    except Exception as e:
        return f"Error: Unable to calculate - {str(e)}"


# Initialize the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Initialize tools
search_tool = TavilySearch(max_results=2)
calc_tool = calculator

tools = [search_tool, calc_tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> Dict[str, List[BaseMessage]]:
    """
    Main chatbot node that processes messages and decides whether to use tools.
    
    Args:
        state: Current state containing conversation messages
        
    Returns:
        Updated state with new message from the LLM
    """
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
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")

# Compile the graph
graph = graph_builder.compile()

# Export for evaluation script
compiled_graph = graph

