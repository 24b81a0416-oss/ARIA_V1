from config import get_hello_name
import os
from unittest.mock import patch
from dotenv import load_dotenv

load_dotenv()

def test_get_hello_name():
    assert get_hello_name() == "World"

@patch.dict(os.environ, {"HELLO_NAME": "Bob"})
def test_get_hello_name_override():
    assert get_hello_name() == "Bob"