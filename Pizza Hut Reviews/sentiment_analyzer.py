#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# sentiment_analyzer.py
from transformers import pipeline

def analyze_sentiment(df):
    """
    Classify the sentiment of each review in the DataFrame.
    Adds a 'predicted_sentiment' column with values 'positive', 'neutral', or 'negative'.
    """
    try:
        classifier = pipeline("sentiment-analysis",
                              model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    sentiments = []
    for text in df['text']:
        try:
            # Truncate long text for model input
            result = classifier(text[:512])[0]
            label = result['label']      # 'POSITIVE' or 'NEGATIVE'
            score = result['score']      # confidence
            # Map to sentiment categories; use a threshold for neutrality
            if score < 0.6:
                sentiments.append("neutral")
            else:
                sentiments.append("positive" if label == "POSITIVE" else "negative")
        except Exception as e:
            sentiments.append("neutral")
            print(f"Error during sentiment analysis: {e}")

    df['predicted_sentiment'] = sentiments
    return df

