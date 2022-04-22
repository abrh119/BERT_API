import numpy as np 
import pandas as pd 
import tensorflow as tf
import transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel
from tensorflow import keras
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input

model_name = 'bert-base-uncased'

# Max length of tokens
max_length = 128 # max 512

# Load transformers config and set output_hidden_states to False
config = BertConfig.from_pretrained(model_name)
#config.output_hidden_states = False

# Load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
bert = TFAutoModel.from_pretrained(model_name)

test = ["Whoever's willing to fuck you is just too lazy to jerk off"]

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


#model = keras.models.load_model('../input/working-bert-with-classes/Bert_Dcnn_model/saved_model.pb')
new_model = load_model('./Bert_Dcnn_model')
results = new_model.predict(tokenization(test),batch_size=32)
print(results)