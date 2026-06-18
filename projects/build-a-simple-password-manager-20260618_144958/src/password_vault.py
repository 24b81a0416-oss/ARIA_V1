from src.database import Database
from src.models import Password
from src.encryption import Encryption

class PasswordVault:
    """Password vault"""
    def __init__(self):
        self.database = Database()
        self.encryption = Encryption()

    def store_password(self, user_id: int, service: str, password: str):
        """Store a password"""
        session = self.database.get_session()
        encrypted_password = self.encryption.encrypt(password)
        new_password = Password(user_id=user_id, service=service, password=encrypted_password)
        session.add(new_password)
        session.commit()
        session.close()

    def retrieve_password(self, user_id: int, service: str) -> str:
        """Retrieve a password"""
        session = self.database.get_session()
        password = session.query(Password).filter_by(user_id=user_id, service=service).first()
        if password:
            decrypted_password = self.encryption.decrypt(password.password)
            session.close()
            return decrypted_password
        session.close()
        return None