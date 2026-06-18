from src.database import Database
from src.models import User
from src.password_vault import PasswordVault

class Authentication:
    """Authentication services"""
    def __init__(self):
        self.database = Database()
        self.password_vault = PasswordVault()

    def register(self, username: str, password: str):
        """Register a new user"""
        session = self.database.get_session()
        new_user = User(username=username, password=password)
        session.add(new_user)
        session.commit()
        session.close()

    def login(self, username: str, password: str) -> bool:
        """Login a user"""
        session = self.database.get_session()
        user = session.query(User).filter_by(username=username).first()
        if user and user.password == password:
            session.close()
            return True
        session.close()
        return False