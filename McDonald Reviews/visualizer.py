#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

def plot_sentiment_distribution(sentiment_counts, output_file='sentiment_distribution.png'):
    """
    Plot bar chart of sentiment categories.
    """
    categories = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())

    plt.figure(figsize=(6,4))
    plt.bar(categories, counts, color=['red', 'gray', 'green'])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_sentiment_trend(df, output_file='sentiment_trend.png'):
    """
    Plot average sentiment trend over time (positive=1, neutral=0, negative=-1).
    """
    mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_score'] = df['sentiment'].map(mapping)

    # Resample by week
    trend = df.set_index('date').resample('W')['sentiment_score'].mean().dropna()

    plt.figure(figsize=(8,4))
    plt.plot(trend.index, trend.values, marker='o')
    plt.title("Weekly Sentiment Trend")
    plt.xlabel("Date")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

