#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from datetime import datetime, timedelta
import re

def parse_relative_time(relative_str):
    """
    Convert relative times like '3 months ago' into absolute datetime.
    If already an absolute date, try parsing directly.
    """
    if pd.isna(relative_str):
        return None

    s = str(relative_str).lower().strip()
    today = datetime.today()

    try:
        # If it's already a date string
        return pd.to_datetime(s)
    except Exception:
        pass

    # Handle "x days/weeks/months/years ago"
    pattern = r'(\d+)\s+(day|week|month|year)s?\s+ago'
    m = re.search(pattern, s)
    if m:
        val, unit = int(m.group(1)), m.group(2)
        if unit == 'day':
            return today - timedelta(days=val)
        elif unit == 'week':
            return today - timedelta(weeks=val)
        elif unit == 'month':
            return today - timedelta(days=30*val)
        elif unit == 'year':
            return today - timedelta(days=365*val)

    # Fallback: return today's date
    return today

def load_data(file_path):
    """
    Load review data from CSV, clean nulls, and parse timestamps.
    """
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        if 'review' not in df.columns:
            raise ValueError("CSV must contain a 'review' column")

        df = df.dropna(subset=['review'])

        # Parse timestamp if exists
        if 'timestamp' in df.columns:
            df['date'] = df['timestamp'].apply(parse_relative_time)
        else:
            df['date'] = pd.NaT

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

