from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config import Config

class Database:
    """Database connection and session management"""
    def __init__(self):
        self.engine = create_engine(f"postgresql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}/{Config.DB_NAME}")
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self):
        """Get a new database session"""
        return self.Session()