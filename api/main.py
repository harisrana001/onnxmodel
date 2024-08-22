from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import os
import pandas as pd

app = FastAPI()


# Utility functions to load GloVe embeddings and mapping
def load_glove_embeddings(filepath):
    embedding_index = {}
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            emb = np.array(values[1:], dtype='float32')
            embedding_index[word] = emb
    return embedding_index


def load_mapping(filepath):
    mapping_df = pd.read_csv(filepath)
    emoticons = mapping_df['emoticons'].tolist()
    return {idx: emoticons[idx] for idx in range(len(emoticons))}


# Load the ONNX model
model_path = os.path.join('model_files', 'model.onnx')
session = ort.InferenceSession(model_path)

# Load the GloVe embeddings and emoji mapping
glove_path = os.path.join('glove.6B.50d.txt')
embedding_index = load_glove_embeddings(glove_path)

mapping_path = os.path.join('model_files', 'Mapping.csv')
mapp = load_mapping(mapping_path)


class TextInput(BaseModel):
    text: str


def preprocess_text(text, max_len=10):
    tokens = text.lower().split()
    embedding_output = np.zeros((1, max_len, 50))

    for i, token in enumerate(tokens):
        if i >= max_len:
            break
        embedding_vector = embedding_index.get(token)
        if embedding_vector is not None:
            embedding_output[0, i] = embedding_vector

    return embedding_output


def process_output(ort_outs):
    probabilities = ort_outs[0][0]
    predicted_class = np.argmax(probabilities)
    predicted_emoji = mapp[predicted_class]
    return predicted_emoji


@app.post("/predict")
def predict_emoji(input: TextInput):
    processed_text = preprocess_text(input.text)
    ort_inputs = {session.get_inputs()[0].name: processed_text.astype(np.float32)}
    ort_outs = session.run(None, ort_inputs)
    predicted_emoji = process_output(ort_outs)
    return {"emoji": predicted_emoji}
