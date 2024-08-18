from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, ValidationError, model_validator
from pydantic.functional_validators import AfterValidator
from typing import Annotated
from typing_extensions import Self
from database import engine, SessionLocal
from sqlalchemy.orm import Session
import models
from datetime import date, datetime
from hashlib import sha3_512
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="http://localhost:5173",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

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
        
class UserDetailsBase(BaseModel):
    user_id : int
    bedrooms: float | None = None
    house_value: float | None = None
    no_of_children: float | None = None
    tot_ppl: float | None = None
    employment_full_time_employee: float | None = None
    employment_part_time_employee: float | None = None
    employment_retired: float | None = None
    employment_self_employed: float | None = None
    employment_student: float | None = None
    employment_unemployeed_seeking_work: float | None = None
    family_structure_1_non_pensioner: float | None = None
    family_structure_all_pensioners: float | None = None
    family_structure_all_students: float | None = None
    family_structure_couple_with_dependent_children: float | None = None
    family_structure_other: float | None = None
    family_structure_single_parent_dependent_children: float | None = None
    savings_just_managing: float | None = None
    savings_saving_a_lot: float | None = None
    savings_saving_little: float | None = None
    savings_using_savings_in_debt: float | None = None
    house_type_bungalow: float | None = None
    house_type_detached_house: float | None = None
    house_type_flat_maisonette: float | None = None
    house_type_semi_detached: float | None = None
    house_type_terraced: float | None = None

class WeatherInfoBase(BaseModel):
    date: date    
    temperatureMax: float
    windBearing: int
    cloudCover: float
    windSpeed: float
    humidity: float
    day_time_minutes: int | None = None
    is_holiday: int | None = None
    season_Fall: float | None = None
    season_Spring: float | None = None
    season_Summer: float | None = None
    season_Winter: float | None = None

class WeatherInfoUpBase(BaseModel):
    date: None = None | date
    temperatureMax: float
    windBearing: int
    cloudCover: float
    windSpeed: float
    humidity: float
    day_time_minutes: int | None = None
    is_holiday: int | None = None
    season_Fall: float | None = None
    season_Spring: float | None = None
    season_Summer: float | None = None
    season_Winter: float | None = None

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





### USER DETAILS ENDPOINTS ###

## POST ##

@app.post("/user_details", status_code=status.HTTP_201_CREATED)
async def create_user_details(user_details : UserDetailsBase, db: db_dependency):
    db_ud = models.UserDetail(**user_details.model_dump())
    db.add(db_ud)
    db.commit()
    return db_ud

## GET ##

@app.get("/user_details", status_code=status.HTTP_200_OK)
async def get_all_user_details_records(db: db_dependency):
    user_details = db.query(models.UserDetail).all()
    if len(user_details) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No user_details records found')
    return user_details

@app.get("/user_details{user_id}", status_code=status.HTTP_200_OK)
async def get_user_details_by_user_id(user_id : int, db:db_dependency):
    user_details = db.query(models.UserDetail).filter(models.UserDetail.user_id == user_id).all()
    if len(user_details) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No user_details records matching this ID found')
    return user_details

## PUT ##

@app.put("/user_details/{user_id}", status_code=status.HTTP_200_OK)
async def update_user(user_id : int, user_details : UserDetailsBase, db: db_dependency):
    db_user_details = db.query(models.UserDetail).filter(models.UserDetail.user_id == [user_id]).first()
    if db_user_details is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The specified user_detail is not found")
    update_data = user_details.model_dump()
    for key, value in update_data.items():
        if value == None:
            setattr(db_user_details, key, 0)
        else:
            setattr(db_user_details, key, value)
    db_user_details.user_id = user_id
    db.commit()
    return db.query(models.UserDetail).filter(models.UserDetail.user_id == [user_id]).first()

## DELETE ##

@app.delete("/user_details/{user_id}", status_code=status.HTTP_200_OK)
async def delete_user_details(user_id : int, db: db_dependency):
    user_details = db.query(models.UserDetail).filter(models.UserDetail.user_id == user_id).first()
    if user_details is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The specified user is not found")
    db.delete(user_details)
    db.commit()

## WEATHER ENDPOINTS ##

## POST ##

@app.post("/weather", status_code=status.HTTP_201_CREATED)
async def create_weather(weather_details: WeatherInfoBase, db: db_dependency):
    db_weather = models.Weather(**weather_details.model_dump())
    db.add(db_weather)
    db.commit()
    return db_weather

## GET ##

@app.get("/weather", status_code=status.HTTP_200_OK)
async def get_all_weather_records(db: db_dependency):
    wd = db.query(models.Weather).all()
    if len(wd) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No weather records found')
    return wd

@app.get("/weather{date}", status_code=status.HTTP_200_OK)
async def get_user_details_by_user_id(date_ : date, db:db_dependency):
    wd = db.query(models.Weather).filter(models.Weather.date == date_).all()
    if len(wd) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No weather records matching this ID found')
    return wd

## PUT ##

@app.put("/weather/{date}", status_code=status.HTTP_200_OK)
async def update_user(date_ : date, weather_details : WeatherInfoUpBase, db: db_dependency):
    season_dict = {1: 'Winter',
               2: 'Winter',
               3: 'Winter', 
               4: 'Spring',
               5: 'Spring',
               6: 'Summer',
               7: 'Summer',
               8: 'Summer',
               9: 'Summer',
               10: 'Fall',
               11: 'Fall',
               12: 'Winter'}
    weather_details.date = date_
    db_wds = db.query(models.Weather).filter(models.Weather.date == date_).first()
    season = list(season_dict.values())[db_wds.date.month - 1]
    if db_wds is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The specified weather data is not found")
    update_data = weather_details.model_dump()
    for key, value in update_data.items():
        if season in key and 'season' in key:
            setattr(db_wds, key, 1)
        elif 'season' in key:
            setattr(db_wds, key, 0)
        else:
            setattr(db_wds, key, value)
    db.commit()
    return db.query(models.Weather).filter(models.Weather.date == date_).first()

### DELETE ###

@app.delete("/weather/{date}", status_code=status.HTTP_200_OK)
async def delete_user_details(date_ : date, db: db_dependency):
    wd = db.query(models.Weather).filter(models.Weather.date == date_).first()
    if wd is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The specified weather record is not found")
    db.delete(wd)
    db.commit()
