from cryptography.fernet import Fernet
from src.config import Config

class Encryption:
    """Encryption services"""
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt(self, data: str) -> str:
        """Encrypt data"""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt(self, data: str) -> str:
        """Decrypt data"""
        return self.cipher.decrypt(data.encode()).decode()