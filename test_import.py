#!/usr/bin/env python3
"""Simple test script to verify agent.py can be imported successfully."""

try:
    import agent
    print("SUCCESS: Agent imported successfully")
    
    # Check if app is exported
    if hasattr(agent, 'app'):
        print("SUCCESS: 'app' attribute found")
    else:
        print("ERROR: 'app' attribute not found")
        
except ImportError as e:
    print(f"ERROR: Import failed - {e}")
except Exception as e:
    print(f"ERROR: Unexpected error - {e}")
