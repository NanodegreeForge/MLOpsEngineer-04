import pickle
import pandas as pd
import pandas.api.types as pdtypes
import pytest
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, train_model

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

def test_train_model():
    """
    Test train_model function from model.py
    """
    df = pd.read_csv("data/census_clean.csv")

    train, test = train_test_split(df, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    model = train_model(X_train, y_train)
    assert model is not None

def test_compute_model_metrics():
    """
    Test compute_model_metrics function from model.py
    """
    df = pd.read_csv("data/census_clean.csv")
    _, df_test = train_test_split(df, test_size=0.20)
    [model, encoder, lb] = pickle.load(open("model/model.pkl", "rb"))

    X_test, y_test, _, _ = process_data(
        X=df_test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    predictions = inference(model, X_test)
    [model, encoder, lb] = pickle.load(open("model/model.pkl", "rb"))
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    assert precision > 0.5

def test_inference():
    """
    Test inference function from model.py
    """
    df = pd.read_csv("data/census_clean.csv")

    _, df_test = train_test_split(df, test_size=0.20)
    [model, encoder, lb] = pickle.load(open("model/model.pkl", "rb"))

    X_test, _, _, _ = process_data(
        X=df_test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    predictions = inference(model, X_test)

    assert len(X_test) == len(predictions) 

