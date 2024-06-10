from fastapi import FastAPI, Form

# STEP 1. import modules
from transformers import pipeline

# STEP 2. create inference instance
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")


app = FastAPI()


@app.post("/text/")
async def text(text: str = Form()):
    # STEP 3. prepare input date
    # STEP 4. inference
    result = classifier(text)
    return {"username": result}