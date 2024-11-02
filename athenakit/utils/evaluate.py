import ast
import operator as op

# Supported operators, add more if needed
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}

def eval_expr(expr, func):
    """
    Safely evaluate a mathematical expression from a string with variable support.

    Args:
        expr (str): The mathematical expression to evaluate.
        func (callable): A function that returns the value of a variable.

    Returns:
        The result of the evaluated expression.

    Raises:
        TypeError: If the expression contains unsupported operations.
    """
    # Replace '^' with '**' for exponentiation 
    # This is because we never use '^' for XOR in math but it may need to be changed
    expr = expr.replace('^', '**')

    def _eval(node):
        if isinstance(node, ast.Num):  # <number>
            return node.n
        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
            left = _eval(node.left)
            right = _eval(node.right)
            operator = operators.get(type(node.op))
            if operator is None:
                raise TypeError(f"Unsupported operator: {node.op}")
            return operator(left, right)
        elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
            operand = _eval(node.operand)
            operator = operators.get(type(node.op))
            if operator is None:
                raise TypeError(f"Unsupported unary operator: {node.op}")
            return operator(operand)
        elif isinstance(node, ast.Name):  # Variable name
            return func(node.id)
        else:
            raise TypeError(f"Unsupported expression: {node}")
    parsed_expr = ast.parse(expr, mode='eval').body
    return _eval(parsed_expr)

# Usage example
if __name__ == "__main__":
    expr = "2 * x + y / 3"
    variables = {"x": 5, "y": 9}
    func = lambda var : variables[var]

    try:
        result = eval_expr(expr, func)
        print(f"{expr} with {variables} = {result}")
    except Exception as e:
        print(f"Error evaluating '{expr}': {e}")
