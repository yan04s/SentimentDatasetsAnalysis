#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import numpy as np

def plot_sentiment_over_time(df, max_ticks=12, out_file="sentiment_trend_quarterly.png"):
    """
    Plot average sentiment score grouped by quarter.
    - df: DataFrame with columns 'review_datetime_utc' and 'sentiment_score'
    - max_ticks: maximum number of x-ticks to show (auto-thins if more)
    """
    df = df.copy()
    # ensure datetime
    df['date'] = pd.to_datetime(df['review_datetime_utc'], errors='coerce')
    # drop rows without valid date or sentiment
    df = df.dropna(subset=['date', 'sentiment_score'])
    # group by quarter (PeriodIndex)
    qtr_series = df.groupby(df['date'].dt.to_period('Q'))['sentiment_score'].mean()
    if qtr_series.empty:
        raise ValueError("No data to plot after grouping by quarter.")
    # x (timestamps at period start) and y values
    x = qtr_series.index.to_timestamp()   # Timestamp object for plotting
    y = qtr_series.values

    # Prepare labels like "2022-Q1"
    labels = [f"{p.year}-Q{p.quarter}" for p in qtr_series.index]

    # Auto-thin ticks if too many
    n = len(x)
    if n > max_ticks:
        step = int(np.ceil(n / max_ticks))
    else:
        step = 1
    tick_positions = x[::step]
    tick_labels = [labels[i] for i in range(0, n, step)]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o', linewidth=1.6)
    plt.title("Average Sentiment Over Time (Quarterly)")
    plt.xlabel("Quarter")
    plt.ylabel("Sentiment Score")
    ax = plt.gca()
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.show()

def plot_sentiment_distribution(df):
    """Bar chart of sentiment label counts."""
    counts = df['sentiment_label'].value_counts()
    plt.figure(figsize=(6,4))
    counts.plot(kind='bar', color=['lightgreen','lightgray','salmon'])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment"); plt.ylabel("Review Count")
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    plt.show()

def plot_top_keywords(df, n=10):
    """Bar chart of top n frequent words in the corpus."""
    from collections import Counter
    all_words = " ".join(df['clean_text']).split()
    common = Counter(all_words).most_common(n)
    words, freqs = zip(*common)
    plt.figure(figsize=(6,4))
    plt.bar(words, freqs, color='steelblue')
    plt.title("Top Keywords in Reviews")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('top_keywords.png')
    plt.show()

