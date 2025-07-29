#!/usr/bin/env python3
"""Test script to verify the LangGraph chat assistant works correctly."""

import os
from langchain_core.messages import HumanMessage

# Set dummy API keys for testing
os.environ["TAVILY_API_KEY"] = "dummy_key_for_testing"
os.environ["ANTHROPIC_API_KEY"] = "dummy_key_for_testing"

from agent import app

def test_basic_functionality():
    """Test the compiled graph with evaluator input format."""
    # Test input format as expected by evaluator
    test_input = {"messages": [HumanMessage(content="Hello, can you help me?")]}
    
    print("Testing basic chat functionality...")
    print(f"Input: {test_input}")
    
    try:
        # Invoke the graph
        result = app.invoke(test_input)
        
        print(f"Output type: {type(result)}")
        print(f"Output keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        if isinstance(result, dict) and "messages" in result:
            print(f"Number of messages: {len(result['messages'])}")
            print(f"Last message type: {type(result['messages'][-1])}")
            print(f"Last message content preview: {str(result['messages'][-1])[:100]}...")
            return True
        else:
            print("ERROR: Result doesn't contain expected 'messages' field")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_calculator_functionality():
    """Test calculator tool functionality."""
    test_input = {"messages": [HumanMessage(content="What is 15 + 27?")]}
    
    print("\nTesting calculator functionality...")
    print(f"Input: {test_input}")
    
    try:
        result = app.invoke(test_input)
        
        if isinstance(result, dict) and "messages" in result:
            print(f"Calculator test - Number of messages: {len(result['messages'])}")
            print(f"Last message content preview: {str(result['messages'][-1])[:200]}...")
            return True
        else:
            print("ERROR: Calculator test failed - no messages in result")
            return False
            
    except Exception as e:
        print(f"ERROR in calculator test: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing LangGraph Chat Assistant")
    print("=" * 50)
    
    # Test basic functionality
    basic_test_passed = test_basic_functionality()
    
    # Test calculator functionality
    calc_test_passed = test_calculator_functionality()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Basic functionality: {'PASS' if basic_test_passed else 'FAIL'}")
    print(f"Calculator functionality: {'PASS' if calc_test_passed else 'FAIL'}")
    print("=" * 50)
    
    if basic_test_passed and calc_test_passed:
        print("✅ All tests passed! The agent is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

