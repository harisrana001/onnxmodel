from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import os

app = FastAPI()

# Load the ONNX mo
# Assuming `embedding_index` and `mapp` are already loaded
# `embedding_index` is a dictionary mapping words to GloVe vectors
# `mapp` is a dictionary mapping class indices to emojis

class TextInput(BaseModel):
    text: str

def preprocess_text(text, max_len=10):
    tokens = text.lower().split()  # Tokenize and convert to lowercase
    embedding_output = np.zeros((1, max_len, 50))  # Initialize with zeros (1 sample, max_len, 50-dim embeddings)

    for i, token in enumerate(tokens):
        if i >= max_len:
            break  # Truncate to max_len
        embedding_vector = embedding_index.get(token)
        if embedding_vector is not None:
            embedding_output[0, i] = embedding_vector

    return embedding_output

def process_output(ort_outs):
    probabilities = ort_outs[0][0]  # Get the first output, first sample
    predicted_class = np.argmax(probabilities)  # Get the index of the highest probability
    predicted_emoji = mapp[predicted_class]  # Map to emoji
    return predicted_emoji

@app.post("/predict")
def predict_emoji(input: TextInput):
    processed_text = preprocess_text(input.text)
    ort_inputs = {session.get_inputs()[0].name: processed_text.astype(np.float32)}
    ort_outs = session.run(None, ort_inputs)
    predicted_emoji = process_output(ort_outs)
    return {"emoji": predicted_emoji}
