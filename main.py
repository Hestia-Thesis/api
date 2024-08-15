from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, ValidationError, model_validator
from pydantic.functional_validators import AfterValidator
from typing import Annotated
from typing_extensions import Self
from database import engine, SessionLocal
from sqlalchemy.orm import Session
import models
from datetime import date
from hashlib import sha3_512

app = FastAPI()
models.Base.metadata.create_all(bind=engine)


#### PYDANTIC MODELS ####

class UserBase(BaseModel):
    email : str
    password : str

class UserUpdate(BaseModel):
    email : str | None = None
    password: str | None = None


class EnergyBase(BaseModel):
    user_id : int
    month : int
    year : int
    consumed : int
    predicted : int

    @model_validator(mode='after')
    def check_date_range(self) -> Self:
        try:
            given_date = date(year=self.year,month=self.month,day=1)
        except Exception as e:
            raise e
        current_date = date(year = date.today().year, month = date.today().month, day = 1)
        if given_date >= current_date:
            raise ValueError('The given date must in the past')
        minimum_date = date(year= current_date.year-4, month= current_date.month, day=current_date.day)
        if given_date < minimum_date:
            raise ValueError(f'The given date cannot be longer than 4 years ago: minimum month is {minimum_date.month}/{minimum_date.year}')
        return self

class EnergyUpdate(BaseModel):
    month : int | None = None
    year : int | None = None
    consumed : int | None = None
    predicted : int | None = None

    @model_validator(mode='after')
    def check_date_range(self) -> Self:
        try:
            given_date = date(year=self.year,month=self.month,day=1)
        except Exception as e:
            raise e
        current_date = date(year = date.today().year, month = date.today().month, day = 1)
        if given_date >= current_date:
            raise ValueError('The given date must in the past')
        minimum_date = date(year= current_date.year-4, month= current_date.month, day=current_date.day)
        if given_date < minimum_date:
            raise ValueError(f'The given date cannot be longer than 4 years ago: minimum month is {minimum_date.month}/{minimum_date.year}')
        return self
        

def get_db():
    db = SessionLocal()
    try:
        yield db
    except:
        raise 
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

#### ENDPOINTS #####

### USER ENDPOINTS ###

## POST ##
@app.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(user: UserBase, db: db_dependency):
    user.password = sha3_512(user.password.encode("utf-8"), usedforsecurity=True).hexdigest()
    db_user = models.User(**user.model_dump())
    db.add(db_user)
    db.commit()

## GET ##
@app.get("/users", status_code=status.HTTP_200_OK)
async def get_all_users(db: db_dependency):
    users = db.query(models.User).all()
    if len(users) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No users found')
    return users

@app.get("/users/{user_id}", status_code=status.HTTP_200_OK)
async def get_user_by_id(user_id : int, db: db_dependency):
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No user matching this ID found")
    return user

## PUT ##
@app.put("/users/{user_id}", status_code=status.HTTP_200_OK)
async def update_user(user_id : int, user : UserUpdate, db: db_dependency):
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No user matching this ID found")
    if user.email is not None:
        db_user.email = user.email
    if user.password is not None:
        db_user.password = user.password    
    db.commit()
    return user

## DELETE ##
@app.delete("/users/{user_id}", status_code=status.HTTP_200_OK)
async def delete_user(user_id : int, db : db_dependency):
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No user matching this ID found")
    db.delete(user)
    db.commit()


### ENERGY ENDPOINTS ###

## POST ##
@app.post("/energy", status_code=status.HTTP_201_CREATED)
async def add_energy_consumption(energy : EnergyBase, db: db_dependency):
    db_energy = models.Energy(**energy.model_dump())
    db.add(db_energy)
    db.commit()

## GET ##
@app.get("/energy", status_code=status.HTTP_200_OK)
async def get_all_energy_records(db: db_dependency):
    energy = db.query(models.Energy).all()
    if len(energy) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No energy records found')
    return energy

@app.get("/energy{user_id}", status_code=status.HTTP_200_OK)
async def get_energy_by_user_id(user_id : int, db:db_dependency):
    energy = db.query(models.Energy).filter(models.Energy.user_id == user_id).all()
    if len(energy) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No energy records matching this ID found')
    return energy

## PUT ##
@app.put("/energy/{user_id}/{year}/{month}", status_code=status.HTTP_200_OK)
async def update_energy_record(user_id : int, year: int, month: int, energy : EnergyUpdate, db: db_dependency):
    db_energy = db.query(models.Energy).filter(models.Energy.user_id == user_id, models.Energy.year == year, models.Energy.month == month).first()
    if db_energy is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The specified energy record is not found")
    if energy.month is not None:
        db_energy.month = energy.month
    if energy.year is not None:
        db_energy.year = energy.year
    if energy.consumed is not None:
        db_energy.consumed = energy.consumed
    if energy.predicted is not None:
        db_energy.predicted = energy.predicted  
    db.commit()
    return energy

## DELETE
@app.delete("/energy/{user_id}/{year}/{month}", status_code=status.HTTP_200_OK)
async def delete_energy_record(user_id : int, year: int, month: int, db: db_dependency):
    energy = db.query(models.Energy).filter(models.Energy.user_id == user_id, models.Energy.year == year, models.Energy.month == month).first()
    if energy is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The specified energy record is not found")
    db.delete(energy)
    db.commit()