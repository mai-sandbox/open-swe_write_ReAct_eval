"""Main LangGraph chat assistant implementation."""

from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import tavily_search, calculator


class State(TypedDict):
    """State definition for the chat assistant."""
    messages: Annotated[list[BaseMessage], add_messages]


def create_agent():
    """Create and configure the LangGraph chat assistant."""
    
    # Initialize Anthropic Claude LLM
    llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
    
    # Define tools
    tools = [tavily_search, calculator]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create StateGraph
    graph_builder = StateGraph(State)
    
    def chatbot(state: State) -> dict:
        """Main chatbot node that processes messages and decides on tool usage."""
        try:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        except Exception as e:
            # Handle LLM errors gracefully
            error_msg = f"I encountered an error: {str(e)}. Please try again."
            from langchain_core.messages import AIMessage
            return {"messages": [AIMessage(content=error_msg)]}
    
    # Add chatbot node
    graph_builder.add_node("chatbot", chatbot)
    
    # Add tool node
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    # Add conditional edges for tool routing
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    
    # Add edge from tools back to chatbot
    graph_builder.add_edge("tools", "chatbot")
    
    # Add entry point
    graph_builder.add_edge(START, "chatbot")
    
    # Compile the graph
    graph = graph_builder.compile()
    
    return graph


def main():
    """Main function to run the chat assistant with user input loop."""
    print("🤖 Chat Assistant started! Type 'quit' or 'exit' to stop.")
    print("You can ask me questions, request web searches, or get help with math calculations.")
    print("-" * 60)
    
    # Create the agent
    agent = create_agent()
    
    while True:
        try:
            user_input = input("
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Create initial state with user message
            initial_state = {"messages": [HumanMessage(content=user_input)]}
            
            # Run the agent
            result = agent.invoke(initial_state)
            
            # Get the last message (assistant's response)
            if result["messages"]:
                assistant_response = result["messages"][-1]
                print(f"🤖 Assistant: {assistant_response.content}")
            else:
                print("🤖 Assistant: I'm sorry, I couldn't generate a response.")
                
        except KeyboardInterrupt:
            print("
            break
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
            print("Please try again.")


if __name__ == "__main__":
    main()

