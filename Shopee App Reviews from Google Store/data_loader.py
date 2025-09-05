#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from nltk.corpus import stopwords
import re

def load_data(file_path):
    """Load CSV or JSON file into DataFrame."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")
        return df
    except Exception as e:
        print(f"Error reading data: {e}")
        return None

def clean_text(text):
    """Basic text cleaning: remove URLs, non-alphanumeric, and stopwords."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    words = text.lower().split()
    filtered = [w for w in words if w not in stopwords.words('english')]
    return " ".join(filtered)

