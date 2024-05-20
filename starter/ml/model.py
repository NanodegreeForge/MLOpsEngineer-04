import pickle
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

from ml.data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression(max_iter=300, random_state=64)
    model.fit(X_train, y_train)
    return model



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def compute_metrics_by_slices(df, cat_features, label):
    """
    Computes the model metrics on slices of the data
    """
    [model, encoder, lb] = pickle.load(open("model/model.pkl", "rb"))

    metrics= pd.DataFrame(columns=('feature', 'percision', 'recall', 'fbeta'))
    for feature in cat_features:
        for category in df[feature].unique():
            fixed_df = df[df[feature] == category]

            x, y, _, _ = process_data(fixed_df, cat_features, label, False, encoder, lb)

            preds = inference(model, x)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            metrics = pd.concat([metrics, pd.DataFrame({'feature': feature, 'percision': precision, 'recall': recall, 'fbeta': fbeta}, index=[0])])

    metrics.to_csv("model/slice_output.txt", index=False)
    return metrics