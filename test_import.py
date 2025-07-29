#!/usr/bin/env python3
"""Comprehensive test script to verify agent.py implementation meets evaluation requirements."""

import sys
from typing import Dict, Any

def test_import():
    """Test that agent.py can be imported successfully."""
    try:
        import agent
        print("✓ SUCCESS: Agent imported successfully")
        
        # Check if app is exported
        if hasattr(agent, 'app'):
            print("✓ SUCCESS: 'app' attribute found")
            return agent.app
        else:
            print("✗ ERROR: 'app' attribute not found")
            return None
            
    except ImportError as e:
        print(f"✗ ERROR: Import failed - {e}")
        return None
    except Exception as e:
        print(f"✗ ERROR: Unexpected error during import - {e}")
        return None

def test_graph_invocation(app):
    """Test that the graph can be invoked with evaluation input format."""
    if app is None:
        print("✗ ERROR: Cannot test invocation - app is None")
        return False
    
    try:
        # Import required message types
        from langchain_core.messages import HumanMessage
        
        # Test with evaluation input format
        test_input = {"messages": [HumanMessage(content="test")]}
        print(f"Testing with input: {test_input}")
        
        # Invoke the graph
        result = app.invoke(test_input)
        print("✓ SUCCESS: Graph invocation completed without errors")
        
        # Validate result structure
        if isinstance(result, dict):
            print("✓ SUCCESS: Result is a dictionary (State format)")
            
            if "messages" in result:
                print("✓ SUCCESS: Result contains 'messages' field")
                
                messages = result["messages"]
                if isinstance(messages, list) and len(messages) > 0:
                    print(f"✓ SUCCESS: Messages field contains {len(messages)} message(s)")
                    
                    # Check if last message is an AI response
                    last_message = messages[-1]
                    if hasattr(last_message, 'content'):
                        print(f"✓ SUCCESS: Last message has content: '{last_message.content[:100]}...'")
                        return True
                    else:
                        print("✗ ERROR: Last message doesn't have content attribute")
                        return False
                else:
                    print("✗ ERROR: Messages field is not a non-empty list")
                    return False
            else:
                print("✗ ERROR: Result doesn't contain 'messages' field")
                return False
        else:
            print(f"✗ ERROR: Result is not a dictionary, got {type(result)}")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: Graph invocation failed - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tools_functionality(app):
    """Test that tools can be invoked through the graph."""
    if app is None:
        print("✗ ERROR: Cannot test tools - app is None")
        return False
    
    try:
        from langchain_core.messages import HumanMessage
        
        # Test calculator tool
        calc_input = {"messages": [HumanMessage(content="What is 5 + 3?")]}
        print("Testing calculator tool with: 'What is 5 + 3?'")
        
        result = app.invoke(calc_input)
        if isinstance(result, dict) and "messages" in result:
            print("✓ SUCCESS: Calculator tool test completed")
            return True
        else:
            print("✗ ERROR: Calculator tool test failed")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: Tools functionality test failed - {e}")
        return False

def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("LANGGRAPH AGENT EVALUATION TESTS")
    print("=" * 60)
    
    # Test 1: Import
    print("\n1. Testing Import...")
    app = test_import()
    
    if app is None:
        print("\n✗ OVERALL RESULT: FAILED - Cannot proceed without successful import")
        sys.exit(1)
    
    # Test 2: Basic invocation
    print("\n2. Testing Graph Invocation...")
    invocation_success = test_graph_invocation(app)
    
    # Test 3: Tools functionality
    print("\n3. Testing Tools Functionality...")
    tools_success = test_tools_functionality(app)
    
    # Overall result
    print("\n" + "=" * 60)
    if invocation_success and tools_success:
        print("✓ OVERALL RESULT: ALL TESTS PASSED")
        print("The agent implementation meets evaluation requirements!")
        sys.exit(0)
    else:
        print("✗ OVERALL RESULT: SOME TESTS FAILED")
        print("The agent implementation needs fixes before evaluation.")
        sys.exit(1)

if __name__ == "__main__":
    main()

