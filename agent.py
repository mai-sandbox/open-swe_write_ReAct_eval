from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

# Define the state with a messages field
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define a simple calculator tool
@tool
def calculator(a: float, b: float, operation: str) -> float:
    """Performs basic arithmetic operations: add, subtract, multiply, divide."""
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b
    else:
        raise ValueError("Invalid operation.")

# Initialize tools
search_tool = TavilySearch(max_results=2)
tools = [search_tool, calculator]

# Initialize the LLM and bind tools
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node function
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph
compiled_graph = graph_builder.compile()

