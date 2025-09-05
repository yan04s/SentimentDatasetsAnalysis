#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load BERT model once when module is imported
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def classify_sentiment(text):
    """
    Classify sentiment using pretrained BERT model.
    Returns: 'negative', 'neutral', 'positive'
    """
    if not isinstance(text, str) or text.strip() == "":
        return 'neutral'

    inputs = tokenizer.encode(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    rating = torch.argmax(probs).item() + 1  # model returns 0-4, so +1

    # Map star rating to sentiment
    if rating in [1, 2]:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:  # 4, 5
        return 'positive'

