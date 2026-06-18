from greeter import greet
from config import get_hello_name

def main() -> None:
    hello_name = get_hello_name()
    print(greet(hello_name))

if __name__ == "__main__":
    main()