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
    assert response.status_code == 200

def test_predict_low():
    """
    Test (/predict) api POST
    """

    response = client.post("/predict", 
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
    assert response.text == '"[0]"'
    assert response.status_code == 200


def test_predict_high():
    """
    Test (/predict) api POST
    """
    response = client.post("/predict", 
        json = {
            "age": 52,
            "workclass": "Private",
            "fnlgt": 209642,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Married-civ-spouse",
            "occupation": "Prof-specialty",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 14084,
            "capital-loss": 0,
            "hours-per-week": 45,
            "native-country": "United-States"
        }

    )
    assert response.text == '"[1]"'
    assert response.status_code == 200