from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model

# Define the state with messages
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the StateGraph
graph_builder = StateGraph(State)

# Initialize the LLM with tools
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# Define the calculator tool
@tool
def calculator(a: float, b: float, operation: str) -> float:
    """Perform basic arithmetic operations."""
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

# Initialize the search tool
search_tool = TavilySearch(max_results=2)

# Bind tools to the LLM
tools = [calculator, search_tool]
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add nodes to the graph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Define conditional edges
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Define edges
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph
compiled_graph = graph_builder.compile()

