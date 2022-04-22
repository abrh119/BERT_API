from gc import callbacks
from urllib import response
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import os
import transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel
from tensorflow import keras
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.callbacks import Callback 
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

class ModelOutput(Callback):
    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        return keys


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
    results = new_model.predict_on_batch(tokenizedValues,batch_size=32, callbacks=[ModelOutput()]) #predict
    return results


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


@app.post("/predict/")
async def root(comment:UserInput):
    text = [UserInput.comment]
    results = makePrediction.predict_on_batch(text) #predict

    return {"prediction": str(results)}
    # print(comment)
    # response = makePrediction(comment.comment)
    # print(response)
    # update_item_encoded = jsonable_encoder(123)
    # newRes = Item(comment=update_item_encoded)
    #return update_item_encoded

@app.get("/")
async def root():
    return {"message": "BERT Boi is up!"}

    
if __name__  == "__main__":
	port = int(os.environ.get('PORT', 5000))
	run(app, host="0.0.0.0", port=port)


# to run 
# python -m uvicorn main:app --reload