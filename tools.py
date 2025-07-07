"""Tools for the chat assistant."""

import ast
import operator
from typing import Any, Dict

from langchain_core.tools import tool
from tavily import TavilyClient


@tool
def tavily_search(query: str) -> str:
    """Search the web for current information using Tavily.
    
    Args:
        query: The search query to look up
        
    Returns:
        Search results as a formatted string
    """
    try:
        # Initialize Tavily client
        client = TavilyClient()
        
        # Perform search
        response = client.search(query=query, max_results=3)
        
        if not response or 'results' not in response:
            return "No search results found."
        
        # Format results
        results = []
        for result in response['results']:
            title = result.get('title', 'No title')
            content = result.get('content', 'No content available')
            url = result.get('url', '')
            
            formatted_result = f"**{title}**
            if url:
                formatted_result += f"
            results.append(formatted_result)
        
        return "
        
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "20 * 0.15", "150 + 50")
        
    Returns:
        The calculated result as a string
    """
    try:
        # Define allowed operations
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        
        def eval_node(node: ast.AST) -> float:
            """Safely evaluate an AST node."""
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return float(node.value)
                else:
                    raise ValueError(f"Unsupported constant type: {type(node.value)}")
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                op_func = allowed_operators.get(type(node.op))
                if op_func is None:
                    raise ValueError(f"Unsupported operation: {type(node.op)}")
                return op_func(left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                op_func = allowed_operators.get(type(node.op))
                if op_func is None:
                    raise ValueError(f"Unsupported unary operation: {type(node.op)}")
                return op_func(operand)
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")
        
        # Parse and evaluate the expression
        tree = ast.parse(expression, mode='eval')
        result = eval_node(tree.body)
        
        # Format the result nicely
        if result == int(result):
            return str(int(result))
        else:
            return f"{result:.10g}"  # Remove trailing zeros
            
    except (SyntaxError, ValueError) as e:
        return f"Invalid expression: {str(e)}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Calculation error: {str(e)}"

