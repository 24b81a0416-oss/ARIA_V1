import secrets
import string

class PasswordGenerator:
    """Password generation"""
    def __init__(self):
        pass

    def generate_password(self, length: int = 12) -> str:
        """Generate a random password"""
        alphabet = string.ascii_letters + string.digits + string.punctuation
        return ''.join(secrets.choice(alphabet) for _ in range(length))