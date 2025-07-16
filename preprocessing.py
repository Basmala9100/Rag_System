# preprocessing.py
import pandas as pd
import nltk
import re

nltk.download("stopwords")
nltk.download("punkt")

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_data(df):
    """
    Generalized function to preprocess any dataframe into text data.
    Each row is converted into a concatenated string of all its features.
    """
    df = df.dropna(how='all')  # Remove empty rows

    texts = []
    for _, row in df.iterrows():
        row_text = " | ".join([f"{col}: {preprocess_text(row[col])}" for col in df.columns if pd.notnull(row[col])])
        texts.append(row_text)

    return texts

def make_chunks(texts, chunk_size=1000, overlap=200):
    chunks = []
    for text in texts:
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
    return chunks
