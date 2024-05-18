import pickle
import pandas as pd
import pandas.api.types as pdtypes
import pytest
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import inference

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

def test_inference():
    """
    Test inference function from model.py
    """
    data = pd.read_csv("data/census_clean.csv")

    _, test_df = train_test_split(data, test_size=0.20)
    [model, encoder, lb] = pickle.load(open("model/lr_model.pkl", "rb"))

    X_test, y_test, _, _ = process_data(
        X=test_df,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    predictions = inference(model, X_test)

    assert len(predictions) == len(X_test)