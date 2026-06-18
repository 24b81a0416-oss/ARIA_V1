import sys
from src.app import app

def cli():
    """Command-line interface"""
    if len(sys.argv) > 1:
        if sys.argv[1] == 'run':
            app.run(debug=True)
        else:
            print('Invalid command')
    else:
        print('Usage: python -m src.cli run')

if __name__ == '__main__':
    cli()