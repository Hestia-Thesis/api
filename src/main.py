from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, ValidationError, model_validator
from pydantic.functional_validators import AfterValidator
from typing import Annotated, get_type_hints
from typing_extensions import Self
from database import engine, SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import and_
import models
from datetime import date, datetime
from hashlib import sha3_512
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
from dataclasses import dataclass, fields
from dotenv import load_dotenv
import os
import pickle
from pathlib import Path
import numpy as np
import base64
import google.generativeai as genai
##pip install google-generativeai
import random

## RECOMMENDATIONS FOR YARNI
# 1. Always return something like the data/json obj or string
# 2. make the error msgs better by adding smthg like:
#   - f"Error: user_id {user_id} not found"
#   add {} dynamic code to make msgs more reading
# 3. when doing @post if the data already exists send a error msg

load_dotenv(dotenv_path='../../.env', verbose=True)
app = FastAPI()
__ml_version__ = '0.1.1'

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
    day: date
    consumed : int
    predicted : int

    @model_validator(mode='after')
    def check_date_range(self) -> Self:
        try:
            # given_date = date(year=self.year,month=self.month,day=1)
            given_date = self.day
        except Exception as e:
            raise e
        # current_date = date(year = date.today().year, month = date.today().month, day = 1)
        current_date = date.today()
        if given_date >= current_date:
            raise ValueError('The given date must in the past')
        minimum_date = date(year= current_date.year-4, month= current_date.month, day=current_date.day)
        if given_date < minimum_date:
            raise ValueError(f'The given date cannot be longer than 4 years ago: minimum month is {minimum_date.month}/{minimum_date.year}')
        return self

class EnergyUpdate(BaseModel):
    # month : int | None = None
    # year : int | None = None
    day: date | None = None
    consumed : int | None = None
    predicted : int | None = None

    @model_validator(mode='after')
    def check_date_range(self) -> Self:
        try:
            # given_date = date(year=self.year,month=self.month,day=1)
            given_date = self.day
        except Exception as e:
            raise e
        # current_date = date(year = date.today().year, month = date.today().month, day = 1)
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

### IMAGE-STORY ENDPOINTS ###
## HELPER FUNCTIONS
def create_image(prompt: str, style: str = 'anime'):
    API_URL = "https://api-inference.huggingface.co/models/prompthero/openjourney-v4"
    headers = {"Authorization": f"Bearer {os.getenv('huggingface_access_token_write')}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    image_bytes = query({
        "inputs": f'{prompt}, in {style} style',
    })

    return image_bytes

def create_story(prompt: str, word_count: int):

    genai.configure(api_key=os.getenv("gemini_api_key"))

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([f'{prompt}, in {word_count} words and less than 20000 characters'])
    story_pm = StoryBase(
        story=response.text
    )
    return response.text
    db_str = models.Stories(**story_pm.model_dump())
    db.add(db_str)
    db.commit()

def get_prompts(percent: int):
    prompt_dict = {
        "significantly_more": {
            "story": [
                "The energy crisis deepened, casting a shadow over the city. The streets, once bustling, now lay empty and silent.",
                "As the energy consumption spiked beyond predictions, a wave of darkness spread across the city. The once vibrant neighborhoods were now desolate.",
                "The overconsumption of energy led to an unsettling quiet. Buildings stood tall, but the lights that once filled them were dimmed.",
            ],
            "picture": [
                "A cityscape under a darkened sky, with flickering streetlights and empty roads, conveying a sense of loss.",
                "An abandoned urban scene, with low energy lighting and overcast skies, creating a somber mood.",
                "A desolate city square, where only a few lights remain on, surrounded by encroaching darkness.",
            ]
        },
        "slightly_more": {
            "story": [
                "The city managed to stay afloat, but the higher-than-expected energy usage caused some concern. The atmosphere was tense, and the people worried about what might come next.",
                "Energy consumption was a bit higher than predicted, causing a slight strain on resources. The city remained functional, but the unease was palpable.",
                "The energy usage exceeded predictions, but only slightly. There was still hope, but the city had to be cautious going forward.",
            ],
            "picture": [
                "A city with dimmed lights and a hint of weariness, as if pushing through a challenging day.",
                "An urban landscape with some areas of reduced lighting, suggesting a slight strain on resources.",
                "A twilight scene where the city is still active, but shadows are creeping in, hinting at underlying concerns.",
            ]
        },
        "normal": {
            "story": [
                "The city's energy consumption matched the predictions, leading to a sense of stability. Life went on as usual, with everything functioning as expected.",
                "Energy usage aligned perfectly with expectations, creating a balanced environment. The city operated smoothly, with no surprises in store.",
                "With energy consumption exactly as predicted, the city maintained its rhythm. There was a calm in the air, as everything proceeded according to plan.",
            ],
            "picture": [
                "A well-organized cityscape, with evenly lit streets and buildings, reflecting a sense of order and stability.",
                "A balanced urban environment, where everything is in harmony, neither too bright nor too dim, showing a city at peace.",
                "A typical day in the city, with steady traffic, clear skies, and a calm atmosphere, indicating normal operations.",
            ]
        },
        "slightly_less": {
            "story": [
                "The energy usage was slightly below predictions, bringing a sense of relief to the city. People felt optimistic about the future, though they knew challenges remained.",
                "With energy consumption just under the predicted levels, the city breathed a little easier. There was a subtle sense of accomplishment in the air.",
                "Energy usage was slightly lower than expected, a small but welcome surprise. The city carried on with a renewed sense of confidence.",
            ],
            "picture": [
                "A well-lit cityscape, with a few areas of softer light, creating a calm and relaxed atmosphere.",
                "An urban environment with bright skies and active streets, reflecting a positive but cautious mood.",
                "A city square under a bright sky, with people moving about, feeling a bit more at ease than before.",
            ]
        },
        "significantly_less": {
            "story": [
                "The city's energy consumption remained in harmony with predictions, allowing life to continue flourishing. Streets were filled with light and laughter.",
                "With energy usage perfectly managed, the city thrived. Parks were green, and the night sky was clear, dotted with stars.",
                "Thanks to efficient energy management, the city continued to glow. People enjoyed the balance, and the atmosphere was lively and hopeful.",
            ],
            "picture": [
                "A bright and bustling city, filled with well-lit streets, lively crowds, and clear skies, signifying a thriving environment.",
                "A vibrant urban park, bathed in sunlight, with people enjoying a day out, surrounded by energy-efficient buildings.",
                "A picturesque city at night, with glowing streetlights, a starry sky, and a sense of peace and prosperity.",
            ]
        }
    }

    if percent > 25:  # Significantly more
        scenario = "significantly_more"
    elif 15 < percent <= 25:  # Slightly more
        scenario = "slightly_more"
    elif -15 <= percent <= 15:  # Normal consumption
        scenario = "normal"
    elif -25 < percent < -15:  # Slightly less
        scenario = "slightly_less"
    else:  # Significantly less
        scenario = "significantly_less"
    return {
        'story_prompt': random.choice(prompt_dict[scenario]["story"]),
        'picture_prompt': random.choice(prompt_dict[scenario]["picture"])
    }

## POST ##

@app.post("/img_story", status_code=status.HTTP_201_CREATED)
async def create_img_story(energy_details: EnergyBase, db: db_dependency, end_date: date = None, style: str = 'anime', word_count: int = 100):
    end_date = end_date or energy_details.day
    
    existing = db.query(models.ImageStories).filter(and_(
        models.ImageStories.date == energy_details.day,
        models.ImageStories.user_id == energy_details.user_id,
        models.ImageStories.end_date == end_date 
    )).all()
    print([existing, energy_details.day, energy_details.user_id, end_date])
    if existing:
        print('here')
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Images and stories for the user_id {energy_details.user_id} for the date range {energy_details.day} - {end_date} already exists."
        )
    ## calculating percentage difference and getting prompts
    percent_difference = ((energy_details.consumed - energy_details.predicted) / energy_details.predicted ) * 100
    prompts = get_prompts(percent_difference)

    ## creating image and story
    img_data = create_image(prompts['picture_prompt'], style=style)
    img = base64.b64encode(img_data).decode('utf-8')
    story = create_story(prompts['story_prompt'], word_count=word_count)

    ## adding it to the db
    img_story = ImageStoriesBase(
        user_id=energy_details.user_id,
        end_date = end_date,
        date = energy_details.day,
        image_data=img, 
        story=story)
    db_img_story = models.ImageStories(**img_story.model_dump())
    db.add(db_img_story)
    db.commit()
    return img_story
    
## GET ##

@app.get("/img_story", status_code=status.HTTP_200_OK)
async def get_all_img_story(db: db_dependency):
    img_story = db.query(models.ImageStories).all()
    if len(img_story) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No images and stories found')
    return img_story

@app.get("/img_story/{user_id}", status_code=status.HTTP_200_OK)
async def get_all_img_story(user_id: int, db: db_dependency):
    img_story = db.query(models.ImageStories).filter(models.ImageStories.user_id == user_id).all()
    if len(img_story) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'No images and stories with user_id: {img_story.user_id} found')
    return img_story

@app.get("/img_story/{date}", status_code=status.HTTP_200_OK)
async def get_all_img_story(date: date, db: db_dependency):
    img_story = db.query(models.ImageStories).filter(models.ImageStories.date == date).all()
    if len(img_story) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'No images and stories with date {img_story.date} found')
    return img_story

@app.get("/img_story/{user_id}/{date}", status_code=status.HTTP_200_OK)
async def get_all_img_story(user_id: int, date: date, db: db_dependency):
    img_story = db.query(models.ImageStories).filter(and_(
        models.ImageStories.date == date,
        models.ImageStories.user_id == user_id
    )).all()
    if len(img_story) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'No images and stories with user_id: {img_story.user_id}, date {img_story.date} found')
    return img_story

@app.get("/img_story/{user_id}/{date}/{end_date}", status_code=status.HTTP_200_OK)
async def get_all_img_story(user_id: int, date: date, end_date: date,db: db_dependency):
    img_story = db.query(models.ImageStories).filter(and_(
        models.ImageStories.date == date,
        models.ImageStories.user_id == user_id,
        models.ImageStories.end_date == end_date
    )).all()
    if len(img_story) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'No images and stories with user_id: {img_story.user_id}, in range {img_story.date} - {img_story.end_date} found')
    return img_story

## DELETE ##
@app.delete("/img_story/{user_id}", status_code=status.HTTP_200_OK)
async def delete_user(img_story : ImageStoriesBase, db : db_dependency):
    records = db.query(models.ImageStories).filter(
            models.ImageStories.user_id == img_story.user_id
        ).all()
    if records is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No record with the user_id: {img_story.user_id} found")
    for record in records:
        db.delete(record)
    db.commit()

@app.delete("/img_story/{user_id}/{date}/{end_date}", status_code=status.HTTP_200_OK)
async def delete_record(img_story : ImageStoriesUpBase, db : db_dependency):
    record = db.query(models.ImageStories).filter(and_(
            models.ImageStories.user_id == img_story.user_id,
            models.ImageStories.date == img_story.date,
            models.ImageStories.end_date == img_story.end_date
        )
        ).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No record with the user_id: {img_story.user_id}, in range {img_story.date} - {img_story.end_date} found")
    db.delete(record)
    db.commit()


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

@app.delete("/users/all_info/{user_id}", status_code=status.HTTP_200_OK)
async def delete_user(user_id : int, db : db_dependency):

    # Energy
    energy_rec = db.query(models.Energy).filter(models.Energy.user_id == user_id).all()
    if energy_rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The specified energy records are not found")
    for record in energy_rec:
        db.delete(record)

    # Img_stories
    records = db.query(models.ImageStories).filter(
        models.ImageStories.user_id == user_id
    ).all()
    if records is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No user with the user_id: {user_id} found")
    for record in records:
        db.delete(record)

    # UserDetail
    user_details = db.query(models.UserDetail).filter(models.UserDetail.user_id == user_id).first()
    if user_details is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"The specified user of user_id: {user_id} is not found")
    db.delete(user_details)
        
    # User
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
    
@app.post("/energy/ml", status_code=status.HTTP_201_CREATED)
async def add_predict_energy_consumption(energy : EnergyBase, db: db_dependency):
    ## checking if a record already exists
    existing = db.query(models.Energy).filter(and_(
        models.Energy.day == energy.day,
        models.Energy.user_id == energy.user_id
        )).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Energy record for user_id {energy.user_id} on {energy.day} already exists."
        )

    ## creating input fields for the model
    # taking inputs
    weather = db.query(models.Weather).filter(models.Weather.date == energy.day).first()
    user_detail = db.query(models.UserDetail).filter(models.UserDetail.user_id == energy.user_id).first()

    # checking if the records exist
    if weather is None:
        add_weather_from_api(WeatherAPIinfo(day=energy.day))
        weather = db.query(models.Weather).filter(models.Weather.date == energy.day).first()

    if user_detail is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User_detail record for user_id {energy.user_id} does not exist."
        )
    
    # taking the input info as dict
    w_dt = WeatherInfoBase.from_orm(weather).model_dump() if weather else {}
    ud_dt = UserDetailsBase.from_orm(user_detail).model_dump() if user_detail else {}
    # joining dicts
    combined_data = {**w_dt, **ud_dt}
    ml_data_types = get_type_hints(MLdata)

    # Filter and cast the combined_data based on the field names and types in MLdata
    filtered_casted_data = {}
    
    field_names = [field.name for field in fields(MLdata)]
    for key in field_names:
        if key in combined_data:
            expected_type = ml_data_types.get(key)
            if expected_type and combined_data[key] is not None:
                try:
                    # Convert value to the expected type
                    filtered_casted_data[key] = expected_type(combined_data[key])
                except (ValueError, TypeError):
                    # Handle the case where casting fails (e.g., wrong data type)
                    filtered_casted_data[key] = None  # or handle in another way
            else:
                # If the value is None or type not found, handle it
                filtered_casted_data[key] = None
        else:
            # Set default value if key not found in combined_data
            filtered_casted_data[key] = None

    # opening the model
    BASE_DIR = Path(__file__).resolve(strict=True).parent

    with open(f'{BASE_DIR}\\random_forest_regressor_{__ml_version__}.pkl', 'rb') as m:
        ml_model = pickle.load(m)

    # Ensure the dict is ready for model prediction
    prediction = ml_model.predict(np.array(list(filtered_casted_data.values())).reshape(1, -1))[0]

    energy.predicted = prediction
    db_energy = models.Energy(**energy.model_dump())
    
    # adding data to db
    db.add(db_energy)
    db.commit()
    return energy

## GET ##
@app.get("/energy", status_code=status.HTTP_200_OK)
async def get_all_energy_records(db: db_dependency):
    energy = db.query(models.Energy).all()
    if len(energy) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='No energy records found')
    return energy

@app.get("/energy/{user_id}", status_code=status.HTTP_200_OK)
async def get_energy_by_user_id(user_id : int, db:db_dependency):
    energy = db.query(models.Energy).filter(models.Energy.user_id == user_id).all()
    if len(energy) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'No energy records matching this ID {user_id} found')
    return energy

## PUT ##

@app.put("/energy/{user_id}/{day}", status_code=status.HTTP_200_OK)
async def update_energy_record(user_id : int, day : date, energy : EnergyUpdate, db: db_dependency):
    energy.day = day
    db_energy = db.query(models.Energy).filter(models.Energy.user_id == user_id, models.Energy.day == day).first()
    if db_energy is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"The specified energy record for user_id: {user_id} and date {day} is not found")
    if energy.day is not None:
        db_energy.day = energy.day
    if energy.consumed is not None:
        db_energy.consumed = energy.consumed
    if energy.predicted is not None:
        db_energy.predicted = energy.predicted  
    db.commit()
    return energy

## DELETE

@app.delete("/energy/{user_id}/{day}", status_code=status.HTTP_200_OK)
async def delete_energy_record(user_id : int, day: date, db: db_dependency):
    energy = db.query(models.Energy).filter(and_(models.Energy.user_id == user_id, models.Energy.day == day)).first()

    if energy is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The specified energy record is not found")
    db.delete(energy)
    db.commit()

@app.delete("/energy/{user_id}", status_code=status.HTTP_200_OK)
async def delete_energy_user(user_id : int, day: date, db: db_dependency):
    energy = db.query(models.Energy).filter(models.Energy.user_id == user_id).all()

    if energy is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The specified energy record is not found")
    
    for record in energy:
        db.delete(record)

    db.commit()

### USER DETAILS ENDPOINTS ###

## POST ##

@app.post("/user_details", status_code=status.HTTP_201_CREATED)
async def create_user_details(user_details : UserDetailsBase, db: db_dependency):
    db_ud = models.UserDetail(**user_details.model_dump())
    existing = db.query(models.UserDetail).filter(models.UserDetail.user_id == user_details.user_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User_details record for user_id {user_details.user_id} already exists."
        )
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'No user_details records matching the ID: {user_id} found')
    return user_details

## PUT ##

@app.put("/user_details/{user_id}", status_code=status.HTTP_200_OK)
async def update_user(user_id : int, user_details : UserDetailsBase, db: db_dependency):
    db_user_details = db.query(models.UserDetail).filter(models.UserDetail.user_id == [user_id]).first()
    if db_user_details is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"The specified user_detail for user_id {user_id} is not found")
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"The specified user of user_id: {user_id} is not found")
    db.delete(user_details)
    db.commit()

## WEATHER ENDPOINTS ##

## POST ##

@app.post("/weather", status_code=status.HTTP_201_CREATED)
async def create_weather(weather_details: WeatherInfoBase, db: db_dependency):
    db_weather = models.Weather(**weather_details.model_dump())
    existing_weather = db.query(models.Weather).filter(models.Weather.date == weather_details.date).first()
    if existing_weather:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Weather record for {weather_details.date} already exists."
        )
    db.add(db_weather)
    db.commit()
    return db_weather

@app.post("/weather/api", status_code=status.HTTP_201_CREATED)
async def add_weather_from_api(info: WeatherAPIinfo, db: db_dependency):
    existing_weather = db.query(models.Weather).filter(models.Weather.date == info.day).first()
    if existing_weather:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Weather record for {info.day} already exists."
        )
    
    response = requests.get(f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{info.city}/{info.day}?unitGroup={info.unit}&key={os.getenv("visualcrossing_api_key")}&include={info.type_data}&lang={info.lang}')
    response_day_details = response.json()['days'][0]
    
    weather_info = WeatherInfoBase(
        date = date.fromisoformat(response_day_details['datetime']),
        temperatureMax = float(response_day_details['tempmax']),
        windBearing = int(response_day_details['winddir']),
        cloudCover = float(response_day_details['cloudcover']),
        windSpeed = float(response_day_details['windspeed']),
        humidity = float(response_day_details['humidity']),
        day_time_minutes = int((pd.to_datetime(response_day_details['sunset']) - pd.to_datetime(response_day_details['sunrise'])).total_seconds() // 60)
    )
    return await create_weather(weather_info, db)

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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'No weather records matching the date: {date_} found')
    return wd

## PUT ##

@app.put("/weather/{date}", status_code=status.HTTP_200_OK)
async def update_user(date_ : date, weather_details : WeatherInfoUpBase, db: db_dependency):
    db_wds = db.query(models.Weather).filter(models.Weather.date == date_).first()
    if db_wds is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"The specified weather data for date {date_} is not found")
    db.commit()
    return db.query(models.Weather).filter(models.Weather.date == date_).first()

### DELETE ###

@app.delete("/weather/{date}", status_code=status.HTTP_200_OK)
async def delete_user_details(date_ : date, db: db_dependency):
    wd = db.query(models.Weather).filter(models.Weather.date == date_).first()
    if wd is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"The specified weather record for date {date_} is not found")
    db.delete(wd)
    db.commit()

