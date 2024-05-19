import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_root():
    """
    Test root (/) api GET
    """
    response = client.get("/")
    assert response.text == '"Hello World"'

def test_predict():
    """
    Test (/predict) api POST
    """

    res = client.post("/predict", 
        json = {
            "age": 28,
            "workclass": "Private",
            "fnlgt": 338409,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Prof-specialty",
            "relationship": "Wife",
            "race": "Black",
            "sex": "Female",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "Cuba"
        }

    )
    assert res.text == '"[0]"'