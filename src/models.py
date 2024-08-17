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

### FOR POSSIBLE TYPES TO USE UNDER THIS COMMENT CHECK IMPORTS FROM 'sqlalchemy'
### EVERYTHING INSIDE [] U NEED TO CHANGE TO UR NEEDS
'''
class Energy(Base):
    __tablename__ = '[exact db tablename]'

    [DB column1] = Column([change to your type], [ForeignKey(User.user_id)] , [primary_key=True])
    [DB column2] = Column([change to your type], [primary_key=True])
    [DB column3] = Column([change to your type], [primary_key=True])
    [DB column4] = Column([change to your type])
    [DB column5] = Column([change to your type], [nullable=True])
'''