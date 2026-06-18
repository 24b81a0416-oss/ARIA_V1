---FILE: requirements.txt | Project dependencies
python-dotenv
---END FILE

---FILE: .env | Environment variables
GREETING_NAME=World
---END FILE

---FILE: config.py | Configuration module
from dotenv import load_dotenv
import os

load_dotenv()

def get_greeting_name() -> str:
    return os.getenv('GREETING_NAME')
---END FILE

---FILE: greeter.py | Greeter module
from config import get_greeting_name

def greet(name: str) -> str:
    return f"Hello, {name}!"
---END FILE

---FILE: main.py | Main entry point
from greeter import greet
from config import get_greeting_name

def main() -> None:
    greeting_name = get_greeting_name()
    print(greet(greeting_name))

if __name__ == "__main__":
    main()
---END FILE

---FILE: tests/test_greeter.py | Unit tests for greeter module
from greeter import greet
import unittest

class TestGreeter(unittest.TestCase):
    def test_greet(self):
        self.assertEqual(greet("World"), "Hello, World!")

if __name__ == "__main__":
    unittest.main()
---END FILE