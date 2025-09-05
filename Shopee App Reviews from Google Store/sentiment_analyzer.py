#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from textblob import TextBlob

def get_sentiment_textblob(text):
    """Return sentiment label using TextBlob polarity."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

# (Optional BERT-based approach using HuggingFace)
# from transformers import pipeline
# classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
# def get_sentiment_bert(text):
#     result = classifier(text[:512])[0]
#     score = int(result['label'].split()[0])
#     return 'positive' if score >=4 else 'neutral' if score==3 else 'negative'

