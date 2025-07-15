"""
Basic chat assistant with LangGraph, search and calculator tools.
"""

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    """State schema for the chat assistant."""
    messages: Annotated[list, add_messages]


# Initialize the StateGraph with the State schema
graph_builder = StateGraph(State)


def chatbot(state: State) -> dict:
    """Basic chatbot node function."""
    # This will be implemented in the next task
    return {"messages": []}


