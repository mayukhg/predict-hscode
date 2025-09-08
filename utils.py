"""
Utility functions for text preprocessing and model operations.
This module contains reusable functions for both training and inference.
"""

import re
import string
from typing import str


def preprocess_text(text: str) -> str:
    """
    Preprocess text by cleaning and normalizing it for machine learning.
    
    Args:
        text (str): Raw text input to be preprocessed
        
    Returns:
        str: Cleaned and normalized text
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters, keep only alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespace and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def validate_hscode_format(hscode: str) -> bool:
    """
    Validate if the HSCode follows the standard format (e.g., "8517.12.30").
    
    Args:
        hscode (str): HSCode to validate
        
    Returns:
        bool: True if format is valid, False otherwise
    """
    # HSCode pattern: 4 digits, dot, 2 digits, dot, 2 digits
    pattern = r'^\d{4}\.\d{2}\.\d{2}$'
    return bool(re.match(pattern, hscode))


def clean_hscode(hscode: str) -> str:
    """
    Clean and standardize HSCode format.
    
    Args:
        hscode (str): Raw HSCode string
        
    Returns:
        str: Cleaned HSCode in standard format
    """
    if not isinstance(hscode, str):
        raise ValueError("HSCode must be a string")
    
    # Remove any non-digit characters except dots
    cleaned = re.sub(r'[^\d.]', '', hscode)
    
    # Ensure proper format
    parts = cleaned.split('.')
    if len(parts) == 3:
        # Format: XXXX.XX.XX
        return f"{parts[0].zfill(4)}.{parts[1].zfill(2)}.{parts[2].zfill(2)}"
    else:
        # If not in expected format, return as is
        return cleaned
