"""Utility functions for movie data processing and filtering."""

import pandas as pd
import re
from difflib import SequenceMatcher

def fuzzy_match(word, target_words, threshold=0.7):
    """Check if word matches any target word with fuzzy matching."""
    if not word or not target_words:
        return False
    
    word_lower = word.lower().strip()
    for target in target_words:
        if not target:
            continue
        target_lower = target.lower().strip()
        similarity = SequenceMatcher(None, word_lower, target_lower).ratio()
        if similarity >= threshold:
            return True
    return False

def extract_runtime_from_text(text):
    """Extract runtime in minutes from text."""
    text = text.lower()
    
    # Pattern for "X hours Y minutes" or "X hours and Y minutes"
    hours_minutes_pattern = r'(\d+)\s*(?:hours?|hrs?)\s*(?:and\s*)?(\d+)\s*(?:minutes?|mins?)'
    match = re.search(hours_minutes_pattern, text)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        return hours * 60 + minutes
    
    # Pattern for just hours
    hours_pattern = r'(\d+)\s*(?:hours?|hrs?)'
    match = re.search(hours_pattern, text)
    if match:
        return int(match.group(1)) * 60
    
    # Pattern for just minutes
    minutes_pattern = r'(\d+)\s*(?:minutes?|mins?)'
    match = re.search(minutes_pattern, text)
    if match:
        return int(match.group(1))
    
    # Special cases
    if 'hour and half' in text or 'hour and a half' in text:
        return 90
    if 'two hours' in text:
        return 120
    
    return None

def format_runtime_display(runtime_minutes):
    """Format runtime for display."""
    if pd.isna(runtime_minutes) or runtime_minutes <= 0:
        return ""
    
    runtime_minutes = int(runtime_minutes)
    hours = runtime_minutes // 60
    minutes = runtime_minutes % 60
    
    if hours > 0 and minutes > 0:
        return f" • {hours}h {minutes}m"
    elif hours > 0:
        return f" • {hours}h"
    else:
        return f" • {minutes}m"

def clean_movie_data(movies_df):
    """Clean and prepare movie data."""
    # Fill missing values
    movies_df['genre'] = movies_df['genre'].fillna('Unknown')
    movies_df['country'] = movies_df['country'].fillna('Unknown')
    movies_df['age_group'] = movies_df['age_group'].fillna('General')
    movies_df['popular'] = movies_df['popular'].fillna(3.0)
    movies_df['released'] = movies_df['released'].fillna(2000)
    movies_df['runtime'] = movies_df['runtime'].fillna(100)
    
    return movies_df

def is_off_topic_query(query):
    """Check if query is off-topic (not about movies)."""
    off_topic_keywords = [
        'weather', 'politics', 'cooking', 'sports', 'news', 
        'health', 'travel', 'music', 'books', 'recipes'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in off_topic_keywords)