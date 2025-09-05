#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data_loader.py
import pandas as pd
import string
import re
from nltk.corpus import stopwords

def load_and_clean_data(filepath):
    """
    Load CSV into a DataFrame, drop empty reviews, and clean text:
    lowercase, remove punctuation, filter stopwords.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Drop rows where 'text' is NaN or empty
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip().astype(bool)]

    # Prepare stopword set
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove stopwords
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words]
        return " ".join(tokens)

    # Apply cleaning
    df['text'] = df['text'].apply(clean_text)
    return df

