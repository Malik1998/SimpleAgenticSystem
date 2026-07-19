from ..decorators import tool

_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
}


@tool(description="Perform a basic arithmetic operation: op in {add, sub, mul, div}.")
def calculator(a: float, b: float, op: str = "add") -> float:
    if op not in _OPS:
        raise ValueError(f"unknown op {op!r}, expected one of {sorted(_OPS)}")
    return _OPS[op](a, b)
