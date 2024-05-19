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

