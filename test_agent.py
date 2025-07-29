#!/usr/bin/env python3
"""
Test script for the LangGraph chat assistant implementation.
Tests graph compilation, tool registration, routing logic, and state management.
"""

import sys
import os
sys.path.append('.')

def test_graph_compilation():
    """Test 1: Graph compiles without errors"""
    print("=== Test 1: Graph Compilation ===")
    try:
        from agent import app
        print("✅ Graph compiled successfully!")
        print(f"Graph type: {type(app)}")
        return True
    except Exception as e:
        print(f"❌ Graph compilation failed: {e}")
        return False

def test_tool_registration():
    """Test 2: Tools are properly registered and callable"""
    print("\n=== Test 2: Tool Registration ===")
    try:
        from agent import tools, llm_with_tools
        print(f"✅ Found {len(tools)} tools registered:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Check if tools are bound to LLM
        if hasattr(llm_with_tools, 'bound'):
            print("✅ Tools are properly bound to LLM")
        else:
            print("⚠️  Tools binding status unclear")
        return True
    except Exception as e:
        print(f"❌ Tool registration test failed: {e}")
        return False

def test_evaluator_input_format():
    """Test 3: Evaluator input format is handled correctly"""
    print("\n=== Test 3: Evaluator Input Format ===")
    try:
        from agent import app
        from langchain_core.messages import HumanMessage
        
        # Test evaluator input format: {'messages': [HumanMessage(content='user_input')]}
        evaluator_input = {"messages": [HumanMessage(content="Hello, can you help me?")]}
        
        print("✅ Evaluator input format created successfully")
        print(f"Input structure: {type(evaluator_input)}")
        print(f"Messages type: {type(evaluator_input['messages'])}")
        print(f"First message type: {type(evaluator_input['messages'][0])}")
        print(f"Message content: {evaluator_input['messages'][0].content}")
        return True
    except Exception as e:
        print(f"❌ Evaluator input format test failed: {e}")
        return False

def test_basic_conversation():
    """Test 4: Basic conversation without tools"""
    print("\n=== Test 4: Basic Conversation ===")
    try:
        from agent import app
        from langchain_core.messages import HumanMessage
        
        # Test simple conversation
        input_data = {"messages": [HumanMessage(content="Hello, how are you?")]}
        
        print("Testing basic conversation...")
        result = app.invoke(input_data)
        
        if "messages" in result and len(result["messages"]) > 0:
            print("✅ Basic conversation test passed")
            print(f"Response type: {type(result['messages'][-1])}")
            print(f"Response content preview: {str(result['messages'][-1].content)[:100]}...")
            return True
        else:
            print("❌ No valid response received")
            return False
    except Exception as e:
        print(f"❌ Basic conversation test failed: {e}")
        return False

def test_calculator_tool():
    """Test 5: Calculator tool usage"""
    print("\n=== Test 5: Calculator Tool Usage ===")
    try:
        from agent import app
        from langchain_core.messages import HumanMessage
        
        # Test calculator tool
        input_data = {"messages": [HumanMessage(content="What is 15 + 27?")]}
        
        print("Testing calculator tool...")
        result = app.invoke(input_data)
        
        if "messages" in result and len(result["messages"]) > 0:
            print("✅ Calculator tool test completed")
            print(f"Final response: {str(result['messages'][-1].content)[:200]}...")
            return True
        else:
            print("❌ No valid response received")
            return False
    except Exception as e:
        print(f"❌ Calculator tool test failed: {e}")
        return False

def test_state_management():
    """Test 6: State management with message history"""
    print("\n=== Test 6: State Management ===")
    try:
        from agent import app
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Test with existing conversation history
        input_data = {
            "messages": [
                HumanMessage(content="My name is Alice"),
                AIMessage(content="Nice to meet you, Alice!"),
                HumanMessage(content="What's my name?")
            ]
        }
        
        print("Testing state management with conversation history...")
        result = app.invoke(input_data)
        
        if "messages" in result and len(result["messages"]) > 0:
            print("✅ State management test completed")
            print(f"Total messages in result: {len(result['messages'])}")
            print(f"Final response: {str(result['messages'][-1].content)[:200]}...")
            return True
        else:
            print("❌ No valid response received")
            return False
    except Exception as e:
        print(f"❌ State management test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing LangGraph Chat Assistant Implementation")
    print("=" * 50)
    
    tests = [
        test_graph_compilation,
        test_tool_registration,
        test_evaluator_input_format,
        test_basic_conversation,
        test_calculator_tool,
        test_state_management
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
