# BERT_API

### Dataset: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge

The proposed idea is to build an AI model that can detect toxic content such as hate speech. This AI model will be created with using BERT capable of detecting hate speech or similar text by phrasal interpretation rather than word-by-word interpretation or even slang terms, the main target is to detect hate speech based off the contexual meaning of the input rather than a word by word detection. 

The end product of the proposing solution will be a well-trained AI neural network model that will be capable of detecting the above-mentioned text. It will be possible to embed this model into a platform or any other software and extend its functionality as REST api using FASTAPI



### The prediction clasess are 
["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

<img width="948" alt="Capture" src="https://user-images.githubusercontent.com/59731843/164798967-6923ad1d-5169-4bc1-92c0-1d3edc750d1f.PNG">

In order to run, Clone the repo, run the databook along with the dataset, save model, and type in "pip install -r requirements" and finally run python -m uvicorn main:app --reload

This model is integrated in a Social Media Web app to detect toxic comments, also the the user inputs are saved in a database in db.py in a mongoDB database so that the user input can be used for further training as a dataset.

#### Social Media Web app: https://github.com/abrh119/KamakNe_Social_Web_App/tree/master
#### Web app Admin Panel: 
#### Use your mongodb username and pass in db.py

### Testing
<img width="368" alt="Capture" src="https://user-images.githubusercontent.com/59731843/172159740-a2b3ea68-db88-4ff5-8acd-ddfa4f839551.PNG">


# Indepth Summary
A hate speech detection model based off BERT
What is BERt? 
BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion developed by google. 

This means it was pretrained on the raw texts only, with no humans labelling because BERT is designed to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context which is perfect for our project

There are 4 main variations of the BERT Model as follows, and in our case we are going with BERT BASE UNCASED
BERT-Base, Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters
So why Uncased? Because uncased comes with accent markers meaning marks over letters can be removed.

With the initially 12 layers of BERT, and each of these 12 layers contains its own encoder and decoder with their own layer weights, 
In Explain in simple forms this is how BERT works, in our case for this model I have customized in a way that there are 4 other layers added to it, so in total there's 16 layers, with 12 default and 4 head classification

In every one of these layers, there is a neural network defined, therefore in every neural network there are multiple nodes having their distinct weightings or values.

And when an input is passed through a neural network then , operations are performed within the nodes and finally the dot product of the output of every node is the output of that layer

So when the tokenized input is passed through the model, the text is passed through each layer, and output of each layer is then held at the end of each layer 

And in the 16th layer we have 6 specified nodes with its own layer weights and each nodes weighting is equivalent to either one of those 6 labels that we have classified already 

## For a simpler diagram based view

![Viva Presentation (1)](https://user-images.githubusercontent.com/59731843/172158506-99671b30-1832-4925-8c89-2a4179a9f807.png)
 

