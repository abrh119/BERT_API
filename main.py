from gc import callbacks
from urllib import response
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from uvicorn.config import LOGGING_CONFIG
import os
from transformers import BertConfig, BertTokenizerFast, TFAutoModel
from tensorflow.python.keras.models import load_model
from pydantic import BaseModel

from typing import List

PORT = int(os.getenv("PORT", 5000))
log_config = LOGGING_CONFIG
log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
# class ModelOutput(Callback):
#     def on_predict_end(self, logs=None):
#         keys = list(logs.keys())
#         print(keys)


model_name = 'bert-base-uncased'
max_length = 128 # max 512

# # Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
bert = TFAutoModel.from_pretrained(model_name)


app = FastAPI()
model_path = './Bert_Dcnn_model/'
new_model = load_model(model_path)


def tokenization (input):
    xy = tokenizer(
    text=list(input),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding='max_length', # padding=True initial value,
     return_tensors='tf',
     return_token_type_ids = False,
     return_attention_mask = True,
     verbose = True)
    return {'input_ids': xy['input_ids'], 'attention_mask': xy['attention_mask']}



async def makePrediction(text):
    if text == "":
        return {"message": "No text provided"}
    tokenizedValues = tokenization(text)
    results = new_model.predict(tokenizedValues,batch_size=32) #predict
    labels=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    class_labels=[labels[i] for i,prob in enumerate(results[0]) if prob >= 0.85 ]
    return class_labels


origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)


class UserInput(BaseModel):
    comment: str

class Response(BaseModel):
    result: List[str] = None


@app.post("/predict/",response_model=Response)
async def root(comment:UserInput):
    text = [comment.comment]
    results = await makePrediction(text) #predict
    res = Response(result = results)
    return res

@app.get("/")
async def root():
    return {"message": "BERT Boi is up!"}

    
if __name__  == "__main__":
	run(app, host="0.0.0.0", port=PORT,log_config=log_config)


# to run 
# python -m uvicorn main:app --reload

# docker image build -t <app-name> .
# docker run -p 5000:5000 -d <app-name>