from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey
from database import Base
import datetime

class User(Base):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, index=True)
    email =  Column(String(50), unique=True)
    password = Column(String(50))

class Energy(Base):
    __tablename__ = 'energy_consumption'

    user_id = Column(Integer, ForeignKey(User.user_id) , primary_key=True)
    month = Column(Integer, primary_key=True)
    year = Column(Integer, primary_key=True)
    consumed = Column(Integer)
    predicted = Column(Integer, nullable=True)