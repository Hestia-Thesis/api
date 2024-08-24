from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, Float, Date, LargeBinary
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, index=True)
    email =  Column(String(50), unique=True)
    password = Column(String(50))

class Energy(Base):
    __tablename__ = 'energy_consumption'

    user_id = Column(Integer, ForeignKey(User.user_id) , primary_key=True)
    day = Column(Date, primary_key=True)
    consumed = Column(Float)
    predicted = Column(Float, nullable=True)

class UserDetail(Base):
    __tablename__ = 'user_details'

    user_id = Column(Integer, ForeignKey(User.user_id), primary_key=True)
    bedrooms = Column(Float, nullable=True)
    house_value = Column(Float, nullable=True)
    no_of_children = Column(Float, nullable=True)
    tot_ppl = Column(Float, nullable=True)
    employment_full_time_employee = Column(Float, nullable=True)
    employment_part_time_employee = Column(Float, nullable=True)
    employment_retired = Column(Float, nullable=True)
    employment_self_employed = Column(Float, nullable=True)
    employment_student = Column(Float, nullable=True)
    employment_unemployeed_seeking_work = Column(Float, nullable=True)
    family_structure_1_non_pensioner = Column(Float, nullable=True)
    family_structure_all_pensioners = Column(Float, nullable=True)
    family_structure_all_students = Column(Float, nullable=True)
    family_structure_couple_with_dependent_children = Column(Float, nullable=True)
    family_structure_other = Column(Float, nullable=True)
    family_structure_single_parent_dependent_children = Column(Float, nullable=True)
    savings_just_managing = Column(Float, nullable=True)
    savings_saving_a_lot = Column(Float, nullable=True)
    savings_saving_little = Column(Float, nullable=True)
    savings_using_savings_in_debt = Column(Float, nullable=True)
    house_type_bungalow = Column(Float, nullable=True)
    house_type_detached_house = Column(Float, nullable=True)
    house_type_flat_maisonette = Column(Float, nullable=True)
    house_type_semi_detached = Column(Float, nullable=True)
    house_type_terraced = Column(Float, nullable=True)

class Weather(Base):
    __tablename__ = 'weather_details'

    date = Column(Date, primary_key=True)
    temperatureMax = Column(Float, nullable=True)
    windBearing = Column(Integer, nullable=True)
    cloudCover = Column(Float, nullable=True)
    windSpeed = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    day_time_minutes = Column(Integer, nullable=True)
    is_holiday = Column(Integer, nullable=True)
    season_Fall = Column(Float, nullable=True)
    season_Spring = Column(Float, nullable=True)
    season_Summer = Column(Float, nullable=True)
    season_Winter = Column(Float, nullable=True)

class ImageStories(Base):
    __tablename__ = 'images_story'
    
    user_id = Column(Integer, ForeignKey(User.user_id), primary_key=True)
    date = Column(Date, primary_key=True)
    end_date = Column(Date, primary_key=True, nullable=True)
    image_data = Column(LargeBinary(length=(2**32)-1))
    story = Column(String(20000))