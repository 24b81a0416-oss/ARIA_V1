from typing import Optional

def generate_greeting(name: Optional[str] = None) -> str:
    """
    Generates a greeting message.

    Args:
    - name (str): The name to include in the greeting. Defaults to None.

    Returns:
    - str: The generated greeting message.
    """
    if name:
        return f"Hello, {name}!"
    else:
        return "Hello, World!"