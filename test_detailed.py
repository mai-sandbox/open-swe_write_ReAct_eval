#!/usr/bin/env python3
"""
Detailed test script to verify the LangGraph implementation functionality.
This script tests the core components without requiring API keys.
"""

import sys
sys.path.append('.')

def test_graph_structure():
    """Test the graph structure and nodes"""
    print("=== Testing Graph Structure ===")
    try:
        from agent import app
        
        # Check graph structure
        print(f"Graph type: {type(app)}")
        print(f"Graph nodes: {list(app.nodes.keys())}")
        # Note: CompiledStateGraph doesn't expose edges directly
        print("Graph compiled successfully with proper structure")
        
        # Verify expected nodes exist
        expected_nodes = ['chatbot', 'tools']
        for node in expected_nodes:
            if node in app.nodes:
                print(f"✅ Node '{node}' found")
            else:
                print(f"❌ Node '{node}' missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Graph structure test failed: {e}")
        return False

def test_tool_functions():
    """Test individual tool functions"""
    print("\n=== Testing Tool Functions ===")
    try:
        from agent import add, subtract, multiply, divide
        
        # Test calculator functions
        test_cases = [
            (add, 5, 3, 8),
            (subtract, 10, 4, 6),
            (multiply, 6, 7, 42),
            (divide, 15, 3, 5)
        ]
        
        for func, a, b, expected in test_cases:
            try:
                result = func.func(a, b)  # Call the underlying function
                if abs(result - expected) < 0.001:
                    print(f"✅ {func.name}({a}, {b}) = {result}")
                else:
                    print(f"❌ {func.name}({a}, {b}) = {result}, expected {expected}")
                    return False
            except Exception as e:
                print(f"❌ {func.name}({a}, {b}) failed: {e}")
                return False
        
        # Test divide by zero
        try:
            divide.func(10, 0)
            print("❌ Divide by zero should raise an error")
            return False
        except ValueError as e:
            print(f"✅ Divide by zero properly handled: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Tool functions test failed: {e}")
        return False

def test_state_schema():
    """Test the State TypedDict schema"""
    print("\n=== Testing State Schema ===")
    try:
        from agent import State
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Test state creation
        test_state = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!")
            ]
        }
        
        print(f"✅ State schema: {State.__annotations__}")
        print(f"✅ Test state created with {len(test_state['messages'])} messages")
        
        # Verify message types
        for i, msg in enumerate(test_state['messages']):
            print(f"  Message {i}: {type(msg).__name__} - {msg.content}")
        
        return True
    except Exception as e:
        print(f"❌ State schema test failed: {e}")
        return False

def test_chatbot_function_structure():
    """Test the chatbot function structure without LLM calls"""
    print("\n=== Testing Chatbot Function Structure ===")
    try:
        from agent import chatbot, State
        from langchain_core.messages import HumanMessage
        
        # Test with empty state (should handle gracefully)
        empty_state = {}
        result = chatbot(empty_state)
        
        if "messages" in result and len(result["messages"]) > 0:
            print("✅ Chatbot handles empty state gracefully")
            print(f"Default response: {result['messages'][0].content}")
        else:
            print("❌ Chatbot failed to handle empty state")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Chatbot function test failed: {e}")
        return False

def test_configuration_files():
    """Test configuration files exist and are valid"""
    print("\n=== Testing Configuration Files ===")
    try:
        import json
        import os
        
        # Check langgraph.json
        if os.path.exists('langgraph.json'):
            with open('langgraph.json', 'r') as f:
                config = json.load(f)
            
            expected_keys = ['dependencies', 'graphs', 'env']
            for key in expected_keys:
                if key in config:
                    print(f"✅ langgraph.json has '{key}': {config[key]}")
                else:
                    print(f"❌ langgraph.json missing '{key}'")
                    return False
        else:
            print("❌ langgraph.json not found")
            return False
        
        # Check agent.py exists
        if os.path.exists('agent.py'):
            print("✅ agent.py exists")
        else:
            print("❌ agent.py not found")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Configuration files test failed: {e}")
        return False

def test_imports_and_dependencies():
    """Test that all required imports work"""
    print("\n=== Testing Imports and Dependencies ===")
    try:
        # Test core imports
        from typing import Annotated, Dict, Any, List
        from typing_extensions import TypedDict
        print("✅ Typing imports successful")
        
        from langchain.chat_models import init_chat_model
        from langchain_core.tools import tool
        from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
        print("✅ LangChain core imports successful")
        
        from langchain_tavily import TavilySearch
        print("✅ Tavily import successful")
        
        from langgraph.graph import StateGraph, START, END
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt import ToolNode, tools_condition
        print("✅ LangGraph imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Imports test failed: {e}")
        return False

def main():
    """Run all detailed tests"""
    print("🔍 Detailed LangGraph Implementation Testing")
    print("=" * 50)
    
    tests = [
        test_imports_and_dependencies,
        test_configuration_files,
        test_graph_structure,
        test_tool_functions,
        test_state_schema,
        test_chatbot_function_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n🏁 Detailed Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All detailed tests passed! Core implementation is solid.")
        print("\n📝 Note: The error messages in the main tests are expected")
        print("   when API keys are not available. The core functionality")
        print("   is working correctly as demonstrated by these detailed tests.")
        return True
    else:
        print("⚠️  Some detailed tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

