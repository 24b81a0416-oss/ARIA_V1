from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from src.database import Database

Base = declarative_base()

class User(Base):
    """User model"""
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

class Password(Base):
    """Password model"""
    __tablename__ = 'passwords'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    service = Column(String, nullable=False)
    password = Column(String, nullable=False)