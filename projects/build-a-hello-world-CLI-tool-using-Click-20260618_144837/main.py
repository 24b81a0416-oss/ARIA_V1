import sys
from src.cli import cli

def main() -> None:
    """
    The main entry point of the application.
    """
    sys.exit(cli())

if __name__ == "__main__":
    main()