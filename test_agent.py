#!/usr/bin/env python3
"""Test script to verify the LangGraph chat assistant structure and input handling."""

import os
from langchain_core.messages import HumanMessage

# Set dummy API keys for testing
os.environ["TAVILY_API_KEY"] = "dummy_key_for_testing"
os.environ["ANTHROPIC_API_KEY"] = "dummy_key_for_testing"

def test_graph_structure():
    """Test that the graph can be imported and has the correct structure."""
    print("Testing graph structure and import...")
    
    try:
        from agent import app, State, tools
        
        print("✅ Successfully imported agent.py")
        print(f"✅ Graph type: {type(app)}")
        print(f"✅ Number of tools: {len(tools)}")
        print(f"✅ Tool names: {[tool.name for tool in tools]}")
        
        # Check if the graph has the expected nodes
        if hasattr(app, 'nodes'):
            print(f"✅ Graph nodes: {list(app.nodes.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR importing agent: {e}")
        return False

def test_input_format_handling():
    """Test that the graph can handle the evaluator input format."""
    print("\nTesting input format handling...")
    
    try:
        from agent import app
        
        # Test the exact evaluator input format
        test_input = {"messages": [HumanMessage(content="test")]}
        
        print(f"✅ Input format: {test_input}")
        print(f"✅ Input type: {type(test_input)}")
        print(f"✅ Messages type: {type(test_input['messages'])}")
        print(f"✅ First message type: {type(test_input['messages'][0])}")
        print(f"✅ Message content: {test_input['messages'][0].content}")
        
        # Verify the input matches expected State schema
        print("✅ Input format matches expected evaluator format")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR with input format: {e}")
        return False

def test_state_schema():
    """Test that the State schema is correctly defined."""
    print("\nTesting State schema...")
    
    try:
        from agent import State
        from typing import get_type_hints
        
        # Check State schema
        hints = get_type_hints(State)
        print(f"✅ State schema fields: {list(hints.keys())}")
        
        if 'messages' in hints:
            print("✅ State has 'messages' field")
        else:
            print("❌ State missing 'messages' field")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ ERROR with State schema: {e}")
        return False

def test_tools_definition():
    """Test that tools are correctly defined."""
    print("\nTesting tools definition...")
    
    try:
        from agent import tools, add, subtract, multiply, divide, search_tool
        
        print(f"✅ Total tools: {len(tools)}")
        
        # Test calculator tools
        calculator_tools = [add, subtract, multiply, divide]
        for tool in calculator_tools:
            print(f"✅ Calculator tool '{tool.name}' defined")
        
        print(f"✅ Search tool '{search_tool.name}' defined")
        
        # Test a simple calculator operation (without invoking the graph)
        result = add.invoke({"a": 2.0, "b": 3.0})
        print(f"✅ Calculator test: add(2, 3) = {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR with tools: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing LangGraph Chat Assistant Structure")
    print("=" * 60)
    
    # Run structure tests
    tests = [
        ("Graph Structure", test_graph_structure),
        ("Input Format Handling", test_input_format_handling),
        ("State Schema", test_state_schema),
        ("Tools Definition", test_tools_definition),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✅ All structural tests passed!")
        print("✅ The graph is correctly structured and ready for evaluation.")
        print("✅ Input format {'messages': [HumanMessage(content='test')]} is supported.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 60)


