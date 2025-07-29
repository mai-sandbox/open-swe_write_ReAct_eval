from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    """State schema for the chat assistant."""
    messages: Annotated[list[BaseMessage], add_messages]


# Initialize the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")  # type: ignore

# Define web search tool
search_tool = TavilySearch(max_results=2)

# Define calculator tools
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first number."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
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


def chatbot(state: State) -> dict:
    """Main chatbot node that processes user messages and decides whether to use tools."""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Create the StateGraph
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




