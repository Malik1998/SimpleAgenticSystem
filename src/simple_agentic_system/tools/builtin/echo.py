from ..decorators import tool


@tool(description="Echo back the given text. Useful for wiring smoke tests.")
def echo(text: str) -> str:
    return text
