from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title = 'Pishing Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)


model = load(pathlib.Path('model/pishing-dataset.joblib'))

class InputData(BaseModel):
    index: int
    having_IPhaving_IP_Address: int
    URLURL_Length: int
    Shortining_Service: int
    having_At_Symbol: int
    double_slash_redirecting: int
    Prefix_Suffix: int
    having_Sub_Domain: int
    SSLfinal_State: int
    Domain_registeration_length: int
    Favicon: int
    port: int
    HTTPS_token: int
    Request_URL: int
    URL_of_Anchor: int
    Links_in_tags: int
    SFH: int
    Submitting_to_email: int
    Abnormal_URL: int
    Redirect: int
    on_mouseover: int
    RightClick: int
    popUpWidnow: int
    Iframe: int
    age_of_domain: int
    DNSRecord: int
    web_traffic: int
    Page_Rank: int
    Google_Index: int
    Links_pointing_to_page: int
    Statistical_report: int



class OutputData(BaseModel):
    score: float
    label: int   # -1 = no phishing, 1 = phishing



FEATURE_ORDER = [
    "index","having_IPhaving_IP_Address","URLURL_Length","Shortining_Service",
    "having_At_Symbol","double_slash_redirecting","Prefix_Suffix","having_Sub_Domain",
    "SSLfinal_State","Domain_registeration_length","Favicon","port","HTTPS_token",
    "Request_URL","URL_of_Anchor","Links_in_tags","SFH","Submitting_to_email",
    "Abnormal_URL","Redirect","on_mouseover","RightClick","popUpWidnow","Iframe",
    "age_of_domain","DNSRecord","web_traffic","Page_Rank","Google_Index",
    "Links_pointing_to_page","Statistical_report"
]

@app.post('/score', response_model=OutputData)
def score(data: InputData):
    vals = [data.dict()[k] for k in FEATURE_ORDER]
    features = np.array([vals], dtype=float)

    pos_idx = int(np.where(model.classes_ == 1)[0][0])
    proba = model.predict_proba(features)[:, pos_idx].item()

    # convertir a etiqueta (-1 o 1)
    label = 1 if proba >= 0.5 else -1

    return {"score": float(proba), "label": label}
