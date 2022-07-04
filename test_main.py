from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["statusCode"] == 200
    assert response.json()["statusText"] == "Root Endpoint for Healthcare Demo API"
    assert response.json()["version"] == "0.0.1-alpha"


def test_wakeup():
    response = client.get("/wakeup")
    assert response.status_code == 200
    assert response.json()["statusCode"] == 200
    assert response.json()["statusText"] == "API Wakeup Successful"
    assert response.json()["version"] == "0.0.1-alpha"


def test_version():
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json()["statusCode"] == 200
    assert response.json()["statusText"] == "Healthcare Demo API Version Fetch Successful"
    assert response.json()["version"] == "0.0.1-alpha"


def test_infer_diabetes():
    response = client.get("/infer/diabetes")
    assert response.status_code == 200
    assert response.json()["statusCode"] == 200
    assert response.json()["statusText"] == "Diabetes Inference Endpoint"
    assert response.json()["version"] == "0.0.1-alpha"


def test_infer_cardiovascular_disease():
    response = client.get("/infer/cardiovascular-disease")
    assert response.status_code == 200
    assert response.json()["statusCode"] == 200
    assert response.json()["statusText"] == "Cardiovascular Disease Inference Endpoint"
    assert response.json()["version"] == "0.0.1-alpha"


def test_infer_pneumonia():
    response = client.get("/infer/pneumonia")
    assert response.status_code == 200
    assert response.json()["statusCode"] == 200
    assert response.json()["statusText"] == "Pneumonia Inference Endpoint"
    assert response.json()["version"] == "0.0.1-alpha"


def test_infer_tuberculosis():
    response = client.get("/infer/tuberculosis")
    assert response.status_code == 200
    assert response.json()["statusCode"] == 200
    assert response.json()["statusText"] == "Tuberculosis Inference Endpoint"
    assert response.json()["version"] == "0.0.1-alpha"


def test_infer_brain_mri():
    response = client.get("/infer/brain-mri")
    assert response.status_code == 200
    assert response.json()["statusCode"] == 200
    assert response.json()["statusText"] == "Brain MRI Inference Endpoint"
    assert response.json()["version"] == "0.0.1-alpha"