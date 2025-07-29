#!/usr/bin/env python3
"""
Test script for the web_search tool to verify it works correctly.
Tests the web search functionality with various query types and error conditions.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Mock the langchain imports to test the web_search function independently
class MockTool:
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def tool(func):
    return MockTool(func)

# Patch the tool decorator
import agent
agent.tool = tool

# Import the web_search function
from agent import web_search

def test_web_search():
    """Test the web_search tool with various queries."""
    
    print("Testing web_search tool:")
    print("=" * 50)
    
    # Test cases with different query types
    test_queries = [
        "python programming",
        "weather forecast",
        "current news",
        "stock market",
        "recipe chocolate cake",
        "health symptoms",
        "machine learning",
        "empty query test",
        "",  # Empty query
        "a" * 250,  # Too long query
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: Query = '{query[:50]}{'...' if len(query) > 50 else ''}'")
        print("-" * 40)
        
        try:
            result = web_search(query)
            
            # Check if result is a string
            if isinstance(result, str):
                print("✓ Returned string result")
                
                # Check if it's an error message
                if result.startswith("Error:"):
                    print(f"✓ Error handled gracefully: {result[:100]}...")
                else:
                    print(f"✓ Search result: {result[:150]}...")
                    
            else:
                print(f"✗ Unexpected result type: {type(result)}")
                
        except Exception as e:
            print(f"✗ Exception occurred: {e}")
    
    print("\n" + "=" * 50)
    print("Web search tool test completed!")

if __name__ == "__main__":
    test_web_search()
