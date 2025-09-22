"""
Utility functions for transcript processing.
"""
import re
from typing import Optional


def extract_video_id(video_url: str) -> Optional[str]:
    """
    Extract video ID from various YouTube URL formats.
    
    Args:
        video_url: YouTube video URL
        
    Returns:
        Video ID or None if invalid URL
    """
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?.*&v=)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)
    return None


def clean_transcript_text(text: str) -> str:
    """
    Clean transcript text by removing duplicates and formatting issues.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned transcript text
    """
    if not text:
        return ""
    
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove music/sound effect annotations
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove timestamp patterns
    text = re.sub(r'\d{1,2}:\d{2}(?::\d{2})?', '', text)
    
    # Clean up multiple spaces created by removals
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def format_time(seconds: float) -> str:
    """
    Convert seconds to readable time format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (HH:MM:SS or MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def validate_transcript(text: str, min_length: int = 100) -> bool:
    """
    Validate if transcript has meaningful content.
    
    Args:
        text: Transcript text
        min_length: Minimum required length
        
    Returns:
        True if transcript is valid
    """
    if not text or len(text) < min_length:
        return False
    
    # Check if it's not just repeated characters or words
    words = text.split()
    if len(set(words)) < 10:  # Less than 10 unique words
        return False
    
    return True