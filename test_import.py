#!/usr/bin/env python3
"""
Test script to verify agent.py can be imported and compiled_graph is exported.

This test verifies that:
1. The agent.py module can be imported without errors
2. The compiled_graph variable is properly exported
3. The implementation works even without external API keys configured
"""

try:
    import agent
    print("✓ Agent imported successfully")
    
    if hasattr(agent, 'compiled_graph'):
        print("✓ compiled_graph attribute exists")
        print(f"✓ compiled_graph type: {type(agent.compiled_graph)}")
        print("✓ All verification checks passed!")
    else:
        print("✗ compiled_graph attribute not found")
        exit(1)
        
except ImportError as e:
    print(f"✗ Failed to import agent: {e}")
    exit(1)
except Exception as e:
    print(f"✗ Error during import: {e}")
    exit(1)

