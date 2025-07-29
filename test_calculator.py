#!/usr/bin/env python3
"""
Test script for the calculator tool to verify it works correctly.
Tests the core calculator logic independently of LangGraph imports.
"""

import ast

def _safe_eval_math_ast(node):
    """
    Safely evaluate an AST node for mathematical expressions.
    
    This function implements security principles similar to ast.literal_eval
    but extends support to basic arithmetic operations (+, -, *, /, **, %).
    Only allows safe mathematical operations on numbers.
    """
    # Handle numeric constants (literals)
    if isinstance(node, ast.Constant):
        # Only allow numeric constants for security
        if isinstance(node.value, (int, float, complex)):
            return node.value
        else:
            return None  # Reject non-numeric literals
    
    # Handle numeric literals (for older Python versions)
    elif isinstance(node, ast.Num):
        return node.n
    
    # Handle binary operations (+, -, *, /, **, %)
    elif isinstance(node, ast.BinOp):
        left = _safe_eval_math_ast(node.left)
        right = _safe_eval_math_ast(node.right)
        
        # Both operands must be valid numbers
        if left is None or right is None:
            return None
        
        # Ensure operands are numeric
        if not isinstance(left, (int, float, complex)) or not isinstance(right, (int, float, complex)):
            return None
            
        # Perform the operation based on the operator type
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("Division by zero")
            return left / right
        elif isinstance(node.op, ast.Pow):
            # Prevent extremely large exponents that could cause system issues
            if isinstance(right, (int, float)) and abs(right) > 1000:
                raise OverflowError("Exponent too large")
            return left ** right
        elif isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("Modulo by zero")
            return left % right
        else:
            return None  # Unsupported binary operation
    
    # Handle unary operations (+x, -x)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_math_ast(node.operand)
        
        if operand is None:
            return None
        
        # Ensure operand is numeric
        if not isinstance(operand, (int, float, complex)):
            return None
            
        if isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.USub):
            return -operand
        else:
            return None  # Unsupported unary operation
    
    # Reject all other node types for security
    else:
        return None


def calculator(expression: str) -> str:
    """
    Test version of calculator function for verification.
    """
    try:
        # Input validation and sanitization
        if not isinstance(expression, str):
            return "Error: Expression must be a string"
        
        expression = expression.strip()
        if not expression:
            return "Error: Empty expression provided"
        
        # Check for potentially dangerous patterns
        dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'file', 'input', 'raw_input']
        if any(pattern in expression.lower() for pattern in dangerous_patterns):
            return f"Error: Expression contains potentially unsafe operations: {expression}"
        
        # Parse the expression into an AST for safe evaluation
        try:
            # Use ast.parse in eval mode to create an AST
            parsed_ast = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            return f"Error: Invalid mathematical syntax in '{expression}': {str(e)}"
        
        # Safely evaluate the AST using our secure evaluator
        result = _safe_eval_math_ast(parsed_ast.body)
        
        if result is None:
            return f"Error: Unsupported operation or invalid expression: {expression}"
        
        # Format the result appropriately
        if isinstance(result, float):
            # Handle very large or very small numbers
            if abs(result) > 1e15:
                return f"Error: Result too large: {result}"
            elif abs(result) < 1e-15 and result != 0:
                return f"Error: Result too small: {result}"
            # Convert integer-valued floats to integers for cleaner display
            elif result.is_integer():
                return str(int(result))
            else:
                # Round to reasonable precision to avoid floating point artifacts
                return str(round(result, 10))
        else:
            return str(result)
            
    except ZeroDivisionError:
        return f"Error: Division by zero in expression: {expression}"
    except OverflowError:
        return f"Error: Number too large to compute in expression: {expression}"
    except ValueError as e:
        return f"Error: Invalid value in expression '{expression}': {str(e)}"
    except Exception as e:
        return f"Error: Failed to calculate '{expression}': {str(e)}"


if __name__ == "__main__":
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

