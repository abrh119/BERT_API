from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import os
from array import array
import transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel
from tensorflow import keras
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input


app = FastAPI()
model_path = './Bert_Dcnn_model'
new_model = load_model(model_path)


def tokenization (input):
    xy = tokenizer(
    text=list([input]),
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
    results = new_model.predict(tokenizedValues,batch_size=32)
    return results


model_name = 'bert-base-uncased'

# Max length of tokens
max_length = 128 # max 512

#Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
config.output_hidden_states = False

#Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
bert = TFAutoModel.from_pretrained(model_name)


# input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
# attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
# inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
# x = bert.bert(inputs)

test = ["Whoever's willing to fuck you is just too lazy to jerk off"]

async def tokenization (input):
    x = await tokenizer(
    text=list(input),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding='max_length', # padding=True initial value,
     return_tensors='tf',
     return_token_type_ids = False,
     return_attention_mask = True,
     verbose = True)
    return {'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']}













async def get_net_image_prediction(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}
    
    img_path = get_file(
        origin = image_link
    )
    img = load_img(
        img_path, 
        target_size = (224, 224)
    )

    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)

    pred = model.predict(img_array)
    score = softmax(pred[0])

    class_prediction = class_predictions[argmax(score)]
    model_score = round(max(score) * 100, 2)

    return {
        "model-prediction": class_prediction,
        "model-prediction-confidence-score": model_score
    }










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

@app.get("/")
async def root():
    return {"message": "BERT Boi!"}
    
if __name__  == "__main__":
	port = int(os.environ.get('PORT', 5000))
	run(app, host="0.0.0.0", port=port)