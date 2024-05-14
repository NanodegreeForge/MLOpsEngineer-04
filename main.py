# Put the code for your API here.
import pickle
import pandas as pd
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}