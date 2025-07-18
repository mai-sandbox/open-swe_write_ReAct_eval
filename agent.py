"""
LangGraph AI Assistant with Search and Calculator Tools

A basic conversational agent that demonstrates core LangGraph concepts including
state management, tool calling, and workflow orchestration.
"""

from typing import Annotated, List
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    """State schema for the LangGraph agent."""
    messages: Annotated[List[BaseMessage], add_messages]


# Initialize the LLM with Anthropic Claude
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
        # Only allow basic arithmetic operations for security
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic arithmetic operations (+, -, *, /, parentheses) and numbers are allowed."
        
        # Evaluate the expression safely
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except Exception as e:
        return f"Error: Could not evaluate the expression '{expression}'. Please check your syntax."


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information.
    
    Args:
        query: The search query string
        
    Returns:
        Search results or error message
    """
    try:
        # Try to use Tavily search if available
        from langchain_tavily import TavilySearch
        
        search_tool = TavilySearch(max_results=3)
        results = search_tool.invoke(query)
        return f"Search results for '{query}':\n{results}"
        
    except ImportError:
        # Fallback if Tavily is not available
        return f"Web search functionality is currently unavailable. I cannot search for '{query}' at this time. Please try asking me something else or provide the information directly."
    except Exception as e:
        return f"Error performing web search for '{query}': {str(e)}"


# Define the tools list
tools = [calculator, web_search]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> dict:
    """
    Main chatbot node that processes messages and decides whether to use tools.
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with new message
    """
    try:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        # Handle LLM errors gracefully
        from langchain_core.messages import AIMessage
        error_message = AIMessage(
            content=f"I apologize, but I encountered an error while processing your request. Please try again."
        )
        return {"messages": [error_message]}


# Create the StateGraph
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("chatbot", chatbot)

# Create tool node for executing tools
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "chatbot")

# Add conditional edges for tool calling
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Tools always route back to chatbot
graph_builder.add_edge("tools", "chatbot")

# Compile the graph
graph = graph_builder.compile()

# Export the compiled graph as required
compiled_graph = graph


