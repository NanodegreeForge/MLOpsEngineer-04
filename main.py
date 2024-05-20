# Put the code for your API here.
import pickle
import pandas as pd
import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()


@app.get("/")
async def root():
    return "Hello World"

class Input(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 20,
                    "workclass": "Private",
                    "fnlgt": 199400,
                    "education": "HS-grad",
                    "education-num": 9,
                    "marital-status": "Married-civ-spouse",
                    "occupation": "Prof-specialty",
                    "relationship": "Wife",
                    "race": "White",
                    "sex": "Female",
                    "capital-gain": 1000,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States"
                }
            ]
        }
    }

@app.post(path="/predict")
async def predict(input: Input):
    [model, encoder, lb] = pickle.load(open("model/model.pkl", "rb"))
    df = pd.DataFrame(input.dict(by_alias=True), index=[0])
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, _, _, _ = process_data(
        X=df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    predictions = inference(model, X)

    return json.dumps(predictions.tolist())