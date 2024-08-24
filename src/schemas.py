from fastapi import HTTPException
from pydantic import BaseModel, ValidationError, model_validator
from typing_extensions import Self
from sqlalchemy.orm import Session
import models
from datetime import date
import requests
from dataclasses import dataclass
#### PYDANTIC MODELS ####

class UserBase(BaseModel):
    email : str
    password : str

class UserUpdate(BaseModel):
    email : str | None = None
    password: str | None = None

class EnergyBase(BaseModel):
    user_id : int
    day: date
    consumed : float
    predicted : float

    @model_validator(mode='after')
    def check_date_range(self) -> Self:
        try:
            given_date = self.day
        except Exception as e:
            raise e
        current_date = date.today()
        if given_date > current_date:
            raise ValueError('The given date must in the past')
        minimum_date = date(year= current_date.year-4, month= current_date.month, day=current_date.day)
        if given_date < minimum_date:
            raise ValueError(f'The given date cannot be longer than 4 years ago: minimum month is {minimum_date.month}/{minimum_date.year}')
        return self

class EnergyUpdate(BaseModel):
    day: date | None = None
    consumed : float | None = None
    predicted : float | None = None

    @model_validator(mode='after')
    def check_date_range(self) -> Self:
        try:
            given_date = self.day
        except Exception as e:
            raise e
        current_date = date.today()
        if given_date is not None:
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

    class Config:
        from_attributes = True

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


    @classmethod
    def get_holidays_list(self):
        # Fetch and process holidays only once
        holidays_response = requests.get('https://www.gov.uk/bank-holidays.json').json()
        holidays_list = []
        for place in holidays_response.keys():
            events = holidays_response[place]["events"]
            holidays_list.extend([event['date'] for event in events if event['date'] not in holidays_list])
        return holidays_list
    
    def get_season(self, month: int):
        # Define the season based on the month
        season_dict = {
            1: 'Winter', 2: 'Winter', 3: 'Winter', 
            4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Summer',
            10: 'Fall', 11: 'Fall', 12: 'Winter'
        }

        season = season_dict[month]

        # Reset all season fields to 0
        for key in ['season_Fall', 'season_Spring', 'season_Summer', 'season_Winter']:
            setattr(self, key, 0)

        # Set the correct season field to 1
        if season == 'Winter':
            self.season_Winter = 1
        elif season == 'Spring':
            self.season_Spring = 1
        elif season == 'Summer':
            self.season_Summer = 1
        elif season == 'Fall':
            self.season_Fall = 1
        
    def __init__(self, **data):
        super().__init__(**data)  # Initialize the Pydantic model first

        # Ensure the holidays list is fetched
        holidays_list = self.get_holidays_list()
        self.is_holiday = 1 if (self.date.strftime("%Y-%m-%d") in holidays_list) or (self.date.weekday() >= 5) else 0

        # Determine the season based on the month
        self.get_season(month=self.date.month)

    class Config:
        from_attributes = True

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

    @model_validator(mode="before")
    def fetch_missing_values(cls, values, **kwargs):
        db: Session = kwargs.get('db')
        date_value = values.get('date')

        # If there is no database session or no date provided, return the values as is
        if db is None or date_value is None:
            return values

        # Fetch the weather info record based on the date
        db_wds = db.query(models.Weather).filter(models.Weather.date == date_value).first()
        if not db_wds:
            raise HTTPException(status_code=404, detail="Weather info not found for the provided date")

        # Fill in missing fields with values from the database
        for field in ['day_time_minutes', 'is_holiday', 'season_Fall', 'season_Spring', 'season_Summer', 'season_Winter']:
            if values.get(field) is None:
                values[field] = getattr(db_wds, field)

        return values

class ImageStoriesUpBase(BaseModel):
    user_id: int
    end_date: date | None = None
    date: date
    image_data: bytes | None = None
    story: str | None = None

class ImageStoriesBase(BaseModel):
    user_id: int
    end_date: date | None = None
    date: date
    image_data: bytes
    story: str

@dataclass
class WeatherAPIinfo:
    day: date
    city: str = "Brussels,Belgium"
    end_date: date | None = None
    unit: str = "metric"
    type_data: str = "days"
    lang: str = 'en'
    
@dataclass
class MLdata:
    temperatureMax: float
    windBearing: int
    cloudCover: float
    windSpeed: float
    humidity: float
    day_time_minutes: int
    is_holiday: int
    season_Fall: float
    season_Spring: float
    season_Summer: float
    season_Winter: float

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
