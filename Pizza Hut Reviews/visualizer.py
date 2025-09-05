#!/usr/bin/env python
# coding: utf-8

# In[1]:


# visualizer.py
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def create_visualizations(df):
    """
    Generate and save:
    1. Bar chart of sentiment distribution.
    2. Pie chart of sentiment distribution.
    3. Word clouds of frequent terms (positive vs. negative).
    """
    # --- Sentiment counts ---
    counts = df['predicted_sentiment'].value_counts()

    # --- Bar chart ---
    plt.figure(figsize=(6,4))
    counts.plot(kind='bar', color=['green','grey','red'])
    plt.title('Sentiment Distribution (Bar Chart)')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.tight_layout()
    plt.savefig('sentiment_distribution_bar.png')
    plt.close()

    # --- Pie chart ---
    plt.figure(figsize=(6,6))
    counts.plot(kind='pie', autopct='%1.1f%%', colors=['green','grey','red'])
    plt.title('Sentiment Distribution (Pie Chart)')
    plt.ylabel('')  # Hide default y-label
    plt.tight_layout()
    plt.savefig('sentiment_distribution_pie.png')
    plt.close()

    # --- Word clouds ---
    pos_text = " ".join(df[df['predicted_sentiment']=='positive']['text'])
    neg_text = " ".join(df[df['predicted_sentiment']=='negative']['text'])
    stopwords = set(STOPWORDS)

    wc_pos = WordCloud(stopwords=stopwords, background_color='white').generate(pos_text)
    wc_neg = WordCloud(stopwords=stopwords, background_color='white').generate(neg_text)

    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    axes[0].imshow(wc_pos)
    axes[0].axis('off')
    axes[0].set_title('Positive Reviews Word Cloud')
    axes[1].imshow(wc_neg)
    axes[1].axis('off')
    axes[1].set_title('Negative Reviews Word Cloud')
    plt.tight_layout()
    fig.savefig('wordcloud.png')
    plt.close()

