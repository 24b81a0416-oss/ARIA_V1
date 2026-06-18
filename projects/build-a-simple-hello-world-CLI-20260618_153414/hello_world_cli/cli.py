<code>
import argparse

def cli():
    """Hello World CLI."""
    parser = argparse.ArgumentParser(description="Prints a hello message.")
    parser.add_argument("--name", help="Name to use in the greeting.")
    args = parser.parse_args()
    
    if args.name:
        print(f"Hello, {args.name}!")
    else:
        print("Hello, World!")
</code>