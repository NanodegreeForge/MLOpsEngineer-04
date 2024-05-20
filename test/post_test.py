import requests

response = requests.post("https://mlopsengineer-04.fly.dev/predict", json = {
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
})

print("Status Code:" + str(response.status_code))
print("Response:" + response.json())
