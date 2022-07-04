import cv2
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from static.utils import do_infer_diabetes, \
                         do_infer_cardiovascular_disease, \
                         decode_image, \
                         encode_image_to_base64, \
                         CFG

VERSION = "0.0.1-alpha"

class Image(BaseModel):
    imageData: str


class DiabetesData(BaseModel):
    pregnancies: int
    plasma_glucose: float
    diastolic_blood_pressure: float
    triceps_thickness: float
    serum_insulin: float
    bmi: float
    diabetes_pedigree: float
    age: int


class CardiovascularData(BaseModel):
    age: int
    gender: int
    height: float
    weight: float
    ap_high: float
    ap_low: float
    cholestrol: float
    glucose: float
    smoke: float
    alcohol: float
    active: float


STATIC_PATH = "static"

origins = [
    "http://localhost:10001",
    "https://pcs-acsq-healthcare-demo-client.netlify.app"
]

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "statusText" : "Root Endpoint for Healthcare Demo API",
        "statusCode" : 200,
        "version" : VERSION,
    }


@app.get("/wakeup")
async def wakeup():
    return {
        "statusCode" : 200,
        "statusText" : "API Wakeup Successful",
        "version" : VERSION,
    } 


@app.get("/version")
async def version():
    return {
        "statusCode" : 200,
        "statusText" : "Healthcare Demo API Version Fetch Successful",
        "version" : VERSION,
    }


@app.get("/infer/diabetes")
async def get_infer_diabetes():
    return {
        "statusCode" : 200,
        "statusText" : "Diabetes Inference Endpoint",
        "version" : VERSION,
    }


@app.get("/infer/cardiovascular-disease")
async def get_infer_cardiovascular_disease():
    return {
        "statusCode" : 200,
        "statusText" : "Cardiovascular Disease Inference Endpoint",
        "version" : VERSION,
    }


@app.get("/infer/pneumonia")
async def get_infer_pneumonia():
    return {
        "statusCode" : 200,
        "statusText" : "Pneumonia Inference Endpoint",
        "version" : VERSION,
    }


@app.get("/infer/tuberculosis")
async def get_infer_tuberculosis():
    return {
        "statusCode" : 200,
        "statusText" : "Tuberculosis Inference Endpoint",
        "version" : VERSION,
    }


@app.get("/infer/brain-mri")
async def get_infer_brain_mri():
    return {
        "statusCode" : 200,
        "statusText" : "Brain MRI Inference Endpoint",
        "version" : VERSION,
    }


@app.post("/infer/diabetes")
async def post_infer_diabetes(data: DiabetesData):
    y_pred, y_pred_proba = do_infer_diabetes([
        data.pregnancies,
        data.plasma_glucose,
        data.diastolic_blood_pressure,
        data.triceps_thickness,
        data.serum_insulin,
        data.bmi,
        data.diabetes_pedigree,
        data.age,
    ])

    if y_pred is not None and y_pred_proba is not None:
        return {
            "statusText": "Diabetes Inference Complete", 
            "statusCode": 200, 
            "prediction": str(y_pred), 
            "probability": str(y_pred_proba),
        }
    else:
        return {
            "statusText" : "Error in performing inference",
            "statusCode" : 404,
        }
    

@app.post("/infer/cardiovascular-disease")
async def post_infer_cardiovascular_disease(data: CardiovascularData):
    y_pred, y_pred_proba = do_infer_cardiovascular_disease([
        data.age,
        data.gender,
        data.height,
        data.weight,
        data.ap_high,
        data.ap_low,
        data.cholestrol,
        data.glucose,
        data.smoke,
        data.alcohol,
        data.active,
    ])

    if y_pred is not None and y_pred_proba is not None:
        return {
            "statusText": "Cardiovascular Disease Inference Complete", 
            "statusCode": 200, 
            "prediction": str(y_pred), 
            "probability": str(y_pred_proba),
        }
    else:
        return {
            "statusText" : "Error in performing inference",
            "statusCode" : 404,
        }
    

@app.post("/infer/pneumonia")
async def post_infer_pneumonia(image: Image):
    _, image = decode_image(image.imageData)
    cfg = CFG(mode="pneumonia")
    cfg.setup()
    
    probability = cfg.infer(image)
    return {
        "statusCode" : 200,
        "statusText" : "Pneumonia Inference Complete",
        "probability" : str(probability),
    }


@app.post("/infer/tuberculosis")
async def post_infer_tuberculosis(image: Image):
    _, image = decode_image(image.imageData)
    cfg = CFG(mode="tuberculosis")
    cfg.setup()
    
    probability = cfg.infer(image)
    return {
        "statusCode" : 200,
        "statusText" : "Tuberculosis Inference Complete",
        "probability" : str(probability),
    }


@app.post("/infer/brain-mri")
async def post_infer_brain_mri(image: Image):
    _, image = decode_image(image.imageData)
    cfg = CFG(mode="brain-abnormality")
    cfg.setup()
    
    image = cfg.infer(image)
    imageData = encode_image_to_base64(image=image)

    return {
        "statusCode" : 200,
        "statusText" : "Brain MRI Inference Complete",
        "imageData" : imageData,
    }
