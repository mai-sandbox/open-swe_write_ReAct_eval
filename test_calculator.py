#!/usr/bin/env python3
"""
Test script for the calculator tool to verify it works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from agent import calculator
    
    print("Testing calculator tool:")
    print("=" * 40)
    
    # Test basic operations
    test_cases = [
        ("2 + 3", "5"),
        ("10 - 4", "6"),
        ("5 * 6", "30"),
        ("15 / 3", "5"),
        ("2 ** 3", "8"),
        ("17 % 5", "2"),
        ("2 + 3 * 4", "14"),  # Order of operations
        ("-5", "-5"),  # Unary minus
        ("+7", "7"),   # Unary plus
    ]
    
    for expression, expected in test_cases:
        result = calculator(expression)
        status = "✓" if result == expected else "✗"
        print(f"{status} {expression} = {result} (expected: {expected})")
    
    print("\nTesting error cases:")
    print("=" * 40)
    
    error_cases = [
        "5 / 0",  # Division by zero
        "2 +",    # Invalid syntax
        "",       # Empty expression
        "import os",  # Dangerous operation
        "2 ** 10000",  # Very large exponent
    ]
    
    for expression in error_cases:
        result = calculator(expression)
        print(f"✓ {expression} -> {result}")
    
    print("\nCalculator tool test completed successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Test error: {e}")
