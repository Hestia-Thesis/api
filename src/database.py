import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from models import Base  


load_dotenv(dotenv_path='../../.env', verbose=True)

DATABASE_URL = os.getenv("DB_URL")

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)

# Create the session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all tables in the database
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()