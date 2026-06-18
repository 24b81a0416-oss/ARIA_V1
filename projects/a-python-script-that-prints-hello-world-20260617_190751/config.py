from dotenv import load_dotenv
from os import environ

load_dotenv()

def get_hello_name() -> str:
    return environ.get('HELLO_NAME', 'World')