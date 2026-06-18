import click
from src.greeting_service import generate_greeting

@click.command()
@click.option("--name", help="The name to include in the greeting.")
def cli(name: str) -> None:
    """
    Prints a greeting message to the console.

    Args:
    - name (str): The name to include in the greeting.
    """
    greeting = generate_greeting(name)
    click.echo(greeting)