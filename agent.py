"""
Basic chat assistant with LangGraph, search and calculator tools.
"""

from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    """State schema for the chat assistant."""
    messages: Annotated[list, add_messages]


# Define the search tool using TavilySearch
search_tool = TavilySearch(max_results=2)


@tool
def calculator(expression: str) -> str:
    """
    Perform basic arithmetic calculations.
    
    Args:
        expression: A mathematical expression string (e.g., "2 + 3", "10 / 2", "5 * 4", "8 - 3")
    
    Returns:
        The result of the calculation as a string, or an error message if the calculation fails.
    """
    try:
        # Parse the expression to extract operands and operator
        expression = expression.strip()
        
        # Handle basic arithmetic operations
        if '+' in expression:
            parts = expression.split('+')
            result = float(parts[0].strip()) + float(parts[1].strip())
        elif '-' in expression:
            parts = expression.split('-')
            result = float(parts[0].strip()) - float(parts[1].strip())
        elif '*' in expression:
            parts = expression.split('*')
            result = float(parts[0].strip()) * float(parts[1].strip())
        elif '/' in expression:
            parts = expression.split('/')
            dividend = float(parts[0].strip())
            divisor = float(parts[1].strip())
            if divisor == 0:
                return "Error: Division by zero is not allowed."
            result = dividend / divisor
        else:
            return f"Error: Unsupported operation in expression '{expression}'. Supported operations: +, -, *, /"
        
        # Return result as string, removing unnecessary decimal places for whole numbers
        if result.is_integer():
            return str(int(result))
        else:
            return str(result)
            
    except ValueError as e:
        return f"Error: Invalid number format in expression '{expression}'. Please use valid numbers."
    except IndexError:
        return f"Error: Invalid expression format '{expression}'. Please use format like '2 + 3'."
    except Exception as e:
        return f"Error: Failed to calculate '{expression}'. {str(e)}"


# List of available tools
tools = [search_tool, calculator]


# Initialize Anthropic Claude LLM and bind tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
llm_with_tools = llm.bind_tools(tools)


# Initialize the StateGraph with the State schema
graph_builder = StateGraph(State)


def chatbot(state: State) -> dict:
    """
    Chatbot node function that processes messages and can use tools when needed.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}



