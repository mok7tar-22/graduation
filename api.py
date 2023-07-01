
import io
from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from tensorflow.python.keras import models
from PIL import Image
import numpy as np
import tensorflow as tf
import uvicorn
import shutil

from fastapi.responses import Response
import os
from random import randint
import uuid
import uvicorn
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

import sklearn
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

diabetes_model    = joblib.load('./models/diabetes_model.sav')
cancer_model      = joblib.load('./models/cancer_model.sav')
heart_model       = joblib.load('./models/heart-model.sav')
chronic_model     = joblib.load('./models/chronic-model.sav')
liver_model       = joblib.load('./models/liver-model.sav')
#malaria_model     =  models.load_model('./models/malaria-model.h5')
#pneumonia_model   =  models.load_model('./models/pneumonia-model.h5')
class DiabetesInput(BaseModel):#8
  Pregnancies:float
  Glucose:float
  BloodPressure:float
  SkinThickness:float
  Insulin:float
  BMI:float
  DiabetesPedigreeFunction:float
  Age:float
class CancerInput(BaseModel):#12
  radius_mean:float
  area_mean:float
  compactness_mean:float      
  concavity_mean:float  
  cpoints_mean:float   
  area_worst:float 
  compactness_worst:float  		 
  concavity_worst:float
  area_se:float
  fractal_dimension_se:float  
  symmetry_worst:float
  fractal_dimension_worst:float

class HeartInput(BaseModel):#13
  age:float
  sex:bool
  cp:float
  trestbps:float
  chol:float
  fbs:float
  restecg:float
  thalach:float
  exang:float
  oldpeak:float
  slope:float
  ca:float
  thal:float

class ChronicInput(BaseModel):#24
  age:float
  blood_pressure:float
  specific_gravity:float
  albumin:float
  sugar:float
  red_blood_cells:float
  pus_cell:float
  pus_cell_clumps:float
  bacteria:float
  blood_glucose_random:float
  blood_urea:float
  serum_creatinine:float
  sodium:float
  potassium:float
  haemoglobin:float
  packed_cell_volume:float
  white_blood_cell_count:float
  red_blood_cell_count:float
  hypertension:float
  diabetes_mellitus:float
  coronary_artery_disease:float
  appetite:float
  peda_edema:float
  aanemia:float
class LiverInput(BaseModel):#6
  age:float
  sex:bool
  total_bilirubin:float
  alkaline_phosphotase:float
  alamine_aminotransferase:float
  albumin_and_globulin_ratio:float

@app.get("/")
async def read_root():
  return {"Hello":"World"}

@app.post("/api/diabetes-disease-predict")
async def diabetes_disease_predict(input:DiabetesInput):
    feature = [input.Pregnancies,input.Glucose,input.BloodPressure,input.SkinThickness,input.Insulin,input.BMI,input.DiabetesPedigreeFunction,input.Age]
    feature = np.array(feature, dtype=float)
    feature = np.asarray(feature)
    feature = feature.reshape(1,-1)
    result  = diabetes_model.predict(feature)
    print(result)
    if (result[0]== 0):
        prediction = 'The Person does not have a Diabetes Disease'
    else:
        prediction='The Person has Diabetes Disease'
    return {"prediction":prediction}

#===============================================================================================
@app.post('/api/cancer-disease-predict')
async def cancer_disease_predict(input:CancerInput):
    feature = [
        input.radius_mean,input.area_mean,input.compactness_mean,input.concavity_mean,
        input.cpoints_mean,input.area_worst,input.compactness_worst,input.concavity_worst,
        input.area_se,input.fractal_dimension_se,input.symmetry_worst,input.fractal_dimension_worst
        ]
    feature = np.array(feature, dtype=float)
    feature = np.asarray(feature)
    feature = feature.reshape(1,-1)
    #feature = cancer_scaler.fit_transform(feature)
    result  = cancer_model.predict(feature)
    print(result)
    if (result[0]== 0):
        prediction = 'The Person does not have a BreastCancer Disease'
    else:
        prediction='The Person has BreastCancer Disease'
    return {"prediction":prediction}
#===============================================================================================
@app.post('/api/heart-disease-predict')
async def heart_disease_predict(input:HeartInput):
  feature = [input.age,input.sex,input.cp,input.trestbps,input.chol,input.fbs,input.restecg,input.thalach,input.exang,input.oldpeak,input.slope,input.ca,input.thal]
  feature = np.array(feature, dtype=float)
  feature = np.asarray(feature)
  feature = feature.reshape(1,-1)
  result  = heart_model.predict(feature)
  print(result)
  if (result[0]== 0):
    prediction = 'The Person does not have a Heart Disease'
  else:
    prediction='The Person has Heart Disease'
  return {"prediction":prediction}

#===============================================================================================
@app.post('/api/chronic-disease-predict')
async def chronic_disease_predict(input:ChronicInput):
    feature = [
        input.age,
        input.blood_pressure,
        input.specific_gravity,
        input.albumin,
        input.sugar,
        input.red_blood_cells,
        input.pus_cell,
        input.pus_cell_clumps,
        input.bacteria,
        input.blood_glucose_random,
        input.blood_urea,
        input.serum_creatinine,
        input.sodium,
        input.potassium,
        input.haemoglobin,
        input.packed_cell_volume,
        input.white_blood_cell_count,
        input.red_blood_cell_count,
        input.hypertension,
        input.diabetes_mellitus,
        input.coronary_artery_disease,
        input.appetite,
        input.peda_edema,
        input.aanemia,
        ]
    feature = np.array(feature, dtype=float)
    feature = np.asarray(feature)
    feature = feature.reshape(1,-1)
    result  = chronic_model.predict(feature)
    print(result)
    if (result[0]== 0):
        prediction = 'The Person does not have a Chronic Disease'
    else:
        prediction='The Person has Chronic Disease'
    return {"prediction":prediction}
#===============================================================================================
@app.post('/api/liver-disease-predict')
async def liver_disease_predict(input:LiverInput):
    feature = [
        input.age,
        input.sex,
        input.total_bilirubin,
        input.alkaline_phosphotase,
        input.alamine_aminotransferase,
        input.albumin_and_globulin_ratio
        ]
    feature = np.array(feature, dtype=float)
    feature = np.asarray(feature)
    feature = feature.reshape(1,-1)
    result  = liver_model.predict(feature)
    print(result)
    if (result[0]== 0):
        prediction = 'The Person does not have a Liver Disease'
    else:
        prediction='The Person has Liver Disease'
    return {"prediction":prediction}


#===========================================================


class PredictionResult(BaseModel):
    label: str
    confidence: float

def read_image(file: UploadFile):
    img = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    return img

def preprocess_image(image):
    resized = cv2.resize(image, (50,50)) # Resize, assuming the model expects 224x224 images
    normalized = resized / 255.0 # Normalize, assuming the model expects pixel values in [0, 1]
    batched = np.expand_dims(normalized, axis=0) # Add batch dimension
    return batched

def load_ml_model(model_path: str):
    return load_model(model_path)

def predict(image, model):
    predictions = model.predict(image)
    label_index = np.argmax(predictions)
    confidence = np.max(predictions)
    return label_index, confidence

MODEL_PATH_MALARIA = "./models/malaria_model.h5" # Update this with the path to your model
model_malaria = load_ml_model(MODEL_PATH_MALARIA)

@app.post("/api/malaria", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...)):
    image = read_image(file)
    preprocessed_image = preprocess_image(image)
    label_index, confidence = predict(preprocessed_image, model_malaria)
    result = PredictionResult(label=str(label_index), confidence=float(confidence))
    return result  
#========================================================


def preprocess_image_pneumonia(image):
    resized = cv2.resize(image, (150,150)) # Resize, assuming the model expects 224x224 images
    normalized = resized / 255.0 # Normalize, assuming the model expects pixel values in [0, 1]
    batched = np.expand_dims(normalized, axis=0) # Add batch dimension
    return batched

MODEL_PATH_PNEUMONIA = "./models/pneumonia_model.h5" # Update this with the path to your model
model_pneumonia = load_ml_model(MODEL_PATH_PNEUMONIA)

@app.post("/api/pneumonia", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...)):
    image = read_image(file)
    preprocessed_image = preprocess_image_pneumonia(image)
    label_index, confidence = predict(preprocessed_image, model_pneumonia)
    result = PredictionResult(label=str(label_index), confidence=float(confidence))
    return result
