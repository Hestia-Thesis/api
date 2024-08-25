from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, ValidationError, model_validator
from pydantic.functional_validators import AfterValidator
from typing import Annotated, get_type_hints
from typing_extensions import Self
from database import engine, SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import and_, extract
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
import calendar
import json
from database import get_db
from schemas import *
import torch

## RECOMMENDATIONS FOR YARNI
# 1. Always return something like the data/json obj or string
# 2. make the error msgs better by adding smthg like:
#   - f"Error: user_id {user_id} not found"
#   add {} dynamic code to make msgs more reading
# 3. when doing @post if the data already exists send a error msg

load_dotenv(dotenv_path='../.env', verbose=True)
app = FastAPI()
__ml_version__ = '0.1.1'

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

models.Base.metadata.create_all(bind=engine)


db_dependency = Annotated[Session, Depends(get_db)]

#### ENDPOINTS #####

### IMAGE-STORY ENDPOINTS ###
## HELPER FUNCTIONS
def create_image(prompt: str, style: str = 'anime'):
    API_URL = "https://api-inference.huggingface.co/models/prompthero/openjourney-v4"
    headers = {"Authorization": f"Bearer {os.getenv('huggingface_access_token_write')}"}

    payload = {
        "inputs": f'{prompt}, in {style} style',
    }
    image_bytes = requests.post(API_URL, headers=headers, json=payload).content

    return image_bytes

def create_story(prompt: str, word_count: int):

    genai.configure(api_key=os.getenv("gemini_api_key"))

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([f'Can you make a story on: {prompt}, in {word_count} words and less than 20000 characters'])
    return response.text

def get_prompts(percent: int):
    prompt_dict = {
        "significantly_more": {
            "story": [
                "The energy crisis deepened, casting a shadow over the city. The streets, once bustling, now lay empty and silent due to a complete lack of electricity.",
                "The excessive energy consumption has accelerated climate change, leading to extreme weather events that devastated the city. Rising sea levels, scorching heatwaves, and violent storms has forced many residents to flee their homes.",
                "As the world is almost out of electricity, energy black markets have started to show. Here people are kidnapped and the little charge produced in the brain from the neurons harvested.",
            ],
            "picture": [
                "A cityscape under a night sky, with streetlights with no lights and empty roads, conveying a sense of loss.",
                "A city being hit by a tsunami, there is thunder in the sky, we see the buildings slowly drowning in the ocean.",
                "A person kidnapped and tied to a chair with a machine on their head, hands and legs are tied back. This person looks miserable",
            ]
        },
        "slightly_more": {
            "story": [
                "We are running out of energy, we will not have enough to even turn on the lights if this continues. We need to stand in a line to get electricity token rations which is crazy expensive.",
                "A group of hackers have infiltrated the city's energy infrastructure and stole valuable resources, causing disruptions.",
                "The city is at the brink of energy crisis which is clear due to occassional power outages, we need to do something quick and save ourselves!",
            ],
            "picture": [
                "A house with dimmed lights while the other houses have lights at night.",
                "A hacker with laptop on the table, it is dark everywhere, he has a smile on his face and on his screen we see he is transferring something to himself.",
                "A twilight scene where the city is still active, but shadows are creeping in, hinting at underlying concerns.",
            ]
        },
        "slightly_less": {
            "story": [
                "The city has achieved a delicate balance between energy consumption and sustainability. Residents are mindful of their energy usage, and renewable energy sources are being integrated into the grid.",
                "A group of scientists and engineers are developing groundbreaking energy technologies that is promised to revolutionize the city's energy infrastructure.",
                "Everyone in the city focused on sustainability. They shared knowledge, resources, and best practices for reducing their energy consumption.",
            ],
            "picture": [
                "A cityscape adorned with lush greenery, solar panels, and wind turbines, symbolizing a sustainable future.",
                "A laboratory filled with scientists working on a new energy technology, representing innovation and progress.",
                "A group of 4 people discussing all are smiling or laughing, there is a lot greenary around them.",
            ]
        },
        "normal": {
            "story": [
                "A group of local entrepreneurs have developed innovative energy-saving technologies, which are being adopted by businesses and households throughout the city.",
                "The family has recently moved into a new, energy-efficient home. They have installed solar panels on the roof, replaced their old appliances with energy-saving models, and planted a vegetable garden to reduce their carbon footprint. Their monthly energy bills are significantly lower than their previous home.",
                "The residents have started a community garden. We use solar-powered lights to illuminate the garden at night and collect rainwater for irrigation.",
            ],
            "picture": [
                "A laboratory showcasing a new energy-saving technology, representing the city's commitment to progress.",
                "A house with a garden where there are a lot of vegetable plants, mosses on the walls, it is a sunny day",
                "A lot of people in a big garden planting plants on a sunny day.",
            ]
        },
        "significantly_less": {
            "story": [
                "The city had achieved a state of energy abundance, thanks to a combination of renewable energy sources and efficient practices. Residents enjoyed a high quality of life, free from the worries of energy shortages.",
                "The city had become completely self-sufficient in terms of energy production. It was a shining example of sustainable living, inspiring other cities to follow suit.",
                "The city was a leader in energy innovation, developing cutting-edge technologies that are being adopted by cities around the world.",
            ],
            "picture": [
                "A cityscape in sunlight, tall trees, houses on the trees and with renewable energy sources powering the city's infrastructure.",
                "A city with thriving green spaces, clean air, and happy, healthy residents, representing the ideal of a sustainable future.",
                "A picturesque city at night, with glowing insets everywhere, happy people, a clear sky with a lot of stars that look like diamonds in the sky, a lot of trees.",
            ]
        }
    }

    if percent > 25:  
        scenario = "significantly_more"
    elif 15 < percent <= 25: 
        scenario = "slightly_more"
    elif -15 <= percent <= 15:
        scenario = "normal"
    elif -25 < percent < -15: 
        scenario = "slightly_less"
    else:
        scenario = "significantly_less"
    random_index = random.randint(0, (len(prompt_dict[scenario]["story"]) - 1))
    return {
        'story_prompt': prompt_dict[scenario]["story"][random_index],
        'picture_prompt': prompt_dict[scenario]["picture"][random_index]
    }

## POST ##

@app.post("/img_story", status_code=status.HTTP_201_CREATED)
async def create_img_story(energy_details: EnergyBase, db: db_dependency, end_date: date = None, style: str = 'anime', word_count: int = 1000):
    end_date = end_date or energy_details.day
    
    existing = db.query(models.ImageStories).filter(and_(
        models.ImageStories.date == energy_details.day,
        models.ImageStories.user_id == energy_details.user_id,
        models.ImageStories.end_date == end_date 
    )).all()
    if existing:
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
    try:
        image_data = base64.b64decode(img)
        js =  json.loads(image_data)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not generated")
    except:
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
        torch.cuda.empty_cache()
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
    db.commit()

    # Img_stories
    records = db.query(models.ImageStories).filter(
        models.ImageStories.user_id == user_id
    ).all()
    if records is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No user with the user_id: {user_id} found")
    for record in records:
        db.delete(record)
    db.commit()

    # UserDetail
    user_details = db.query(models.UserDetail).filter(models.UserDetail.user_id == user_id).first()
    if user_details is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"The specified user of user_id: {user_id} is not found")
    db.delete(user_details)
    db.commit()

        
    # User
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No user matching this ID found")
    db.delete(user)

    db.commit()


### ENERGY ENDPOINTS ###
## POST ##

async def get_prediction(energy:EnergyBase, db: db_dependency):
    ## creating input fields for the model
    # taking inputs
    weather = db.query(models.Weather).filter(models.Weather.date == energy.day).first()
    user_detail = db.query(models.UserDetail).filter(models.UserDetail.user_id == energy.user_id).first()
    # checking if the records exist
    if weather is None:
        await add_weather_from_api(WeatherAPIinfo(day=energy.day),db)
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

    # Get the type hints from MLdata 
    ml_data_types = get_type_hints(MLdata)

    # Filter and cast the combined_data based on the field names and types in MLdata
    filtered_casted_data = {}

    # Get the field names from MLdata
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

    # Load the model
    BASE_DIR = Path(__file__).resolve(strict=True).parent
    model_path = BASE_DIR / f'random_forest_regressor_{__ml_version__}.pkl'

    try:
        with open(model_path, 'rb') as m:
            ml_model = pickle.load(m)
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found at {model_path}")

    # Ensure that the DataFrame matches the feature names used during model training
    if hasattr(ml_model, 'feature_names_in_'):
        model_features = ml_model.feature_names_in_
    else:
        raise RuntimeError("The loaded model does not have the feature_names_in_ attribute. "
                        "Ensure it was trained with a DataFrame.")

    # Create a DataFrame with the correct feature names and a single row
    features_df = pd.DataFrame([filtered_casted_data])

    # Reindex the DataFrame to match the training features order and handle missing columns
    features_df = features_df.reindex(columns=model_features)

    # Handle missing values in features_df
    features_df.fillna(0, inplace=True)  # Replace NaNs with 0 or another value

    # Infer the objects' dtype after filling NaNs
    features_df = features_df.infer_objects(copy=False)

    # Predict using the model
    try:
        prediction = ml_model.predict(features_df)[0]
        return float(prediction)
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

@app.post("/energy", status_code=status.HTTP_201_CREATED)
async def add_energy_consumption(energy : EnergyBase, db: db_dependency):
    db_energy = models.Energy(**energy.model_dump())
    db.add(db_energy)
    db.commit()
    
@app.post("/energy/ml/image_story", status_code=status.HTTP_201_CREATED)
async def add_predict_energy_consumption(energy : EnergyBase, db: db_dependency, img_style:str = 'anime', word_count: int = 1000):
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

    predicted_ = await get_prediction(energy, db)
    energy.predicted = float(predicted_)
    db_energy = models.Energy(**energy.model_dump())
    
    # create_img_story(energy_details: EnergyBase, db: db_dependency, end_date: date = None, style: str = 'anime', word_count: int = 100)
    img_tb = await create_img_story(energy_details=energy, db=db, style=img_style, word_count=word_count)
    # adding data to db
    db.add(db_energy)
    db.commit()
    return {
        'energy_table': energy,
        'image_story_table': img_tb
        }

@app.post("/energy/ml/monthly", status_code=status.HTTP_201_CREATED)
async def add_predict_energy_consumption_month(energy : EnergyBase, db: db_dependency):
    _, total_days = calendar.monthrange(energy.day.year, energy.day.month)
    consumption_per_day = energy.consumed / total_days
    energy_records = []
    for day in range(1, total_days+1):
        try:
            existing = db.query(models.Energy).filter(and_(
                models.Energy.day == energy.day,
                models.Energy.user_id == energy.user_id
                )).first()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Energy record for user_id {energy.user_id} on {energy.day} already exists."
                )

            energy_pm = EnergyBase(
                    user_id=energy.user_id,
                    day=energy.day.replace(day=day),
                    consumed=consumption_per_day,
                    predicted=0
                )
            energy_pm.predicted = await get_prediction(energy_pm, db)
            energy_records.append(energy_pm)
        
        except ValidationError as ve:
            raise HTTPException(
                status_code=422,
                detail=f"Validation error for day {day}: {ve.errors()}"
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred for day {day}: {str(e)}"
            )

    # adding data to db
    db_energy = [models.Energy(**record.model_dump()) for record in energy_records]
    db.add_all(db_energy)
    db.commit()
    return energy_records

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

@app.get("/energy/{user_id}/monthly", status_code=status.HTTP_200_OK)
async def get_energy_by_user_id_month(user_id : int, month: int, year: int,  db:db_dependency):
    energy = db.query(models.Energy).filter(and_(
        models.Energy.user_id == user_id,
        extract('year', models.Energy.day) == year,
        extract('month', models.Energy.day) == month
        )
    ).all(),
    if len(energy) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'No energy records matching this ID {user_id} for {month}/{year} found')
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

@app.get("/user_details/{user_id}", status_code=status.HTTP_200_OK)
async def get_user_details_by_user_id(user_id : int, db:db_dependency):
    user_details = db.query(models.UserDetail).filter(models.UserDetail.user_id == user_id).all()
    if len(user_details) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'No user_details records matching the ID: {user_id} found')
    return user_details

## PUT ##

@app.put("/user_details/{user_id}", status_code=status.HTTP_200_OK)
async def update_user(user_id : int, user_details : UserDetailsBase, db: db_dependency):
    db_user_details = db.query(models.UserDetail).filter(models.UserDetail.user_id == user_id).first()
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
    return db.query(models.UserDetail).filter(models.UserDetail.user_id == user_id).first()

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

@app.get("/weather/{date}", status_code=status.HTTP_200_OK)
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

