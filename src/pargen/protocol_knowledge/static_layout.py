"""Resolve only provably static bit positions, without evaluating arbitrary code."""
import ast
import operator
import re


def static_property(nodes, node_id, prop, *, origin=None, active=None):
    """Return an integer or None. ``origin`` gives a local coordinate frame."""
    try:
        node_id = int(node_id)
    except (TypeError, ValueError):
        return None
    if node_id == origin and prop == "bit_start":
        return 0
    key = (node_id, prop)
    active = set(active or ())
    if key in active:
        return None
    active.add(key)
    value = nodes.get(node_id, {}).get(prop)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
    refs = {}

    def replace(match):
        name = f"ref_{len(refs)}"
        refs[name] = static_property(nodes, match[1], match[2], origin=origin, active=active)
        return name

    expr = re.sub(r"\b(\d+)\.(bit_start|size_bits)\b", replace, value)
    if any(v is None for v in refs.values()):
        return None

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id in refs:
            return refs[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
            if type(node.op) in ops:
                return ops[type(node.op)](left, right)
            if isinstance(node.op, (ast.Div, ast.FloorDiv)) and right and left % right == 0:
                return left // right
        raise ValueError("not a static integer expression")

    try:
        return evaluate(ast.parse(expr.strip(), mode="eval").body)
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, RecursionError):
        return None

