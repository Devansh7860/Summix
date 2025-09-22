"""
YouTube Transcript API fetcher module.
"""
import time
from typing import Optional
from utils import clean_transcript_text
import re

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    TranscriptsDisabled = Exception
    print("⚠️  youtube_transcript_api not installed. Install with: pip install youtube-transcript-api")


class APITranscriptFetcher:
    """Fetches transcripts using the YouTube Transcript API."""
    
    def __init__(self):
        self.available = API_AVAILABLE
        self.language_preferences = [
            'en', 'en-US', 'en-GB', 'en-IN', 'en-CA', 'en-AU', 'en-orig', 'en-NZ', 'en-IE', 'en-ZA', 'en-PH', 'en-SG', 
            'hi', 'hi-IN', 'hi-Latn' , 'hi-orig'
        ]
    
    def is_available(self) -> bool:
        """Check if the API is available."""
        return self.available
    
    def fetch_transcript(self, video_id):

        if not self.available:
            return None
     
            # First, try to get transcript directly with language preferences
            # This is the most efficient approach for the latest API version
            
            # Language preference order: English > Hindi > Any
        language_preferences = [
        'en', 'en-US', 'en-GB', 'en-IN', 'en-CA', 'en-AU', 'en-NZ', 'en-IE', 'en-ZA', 'en-PH', 'en-SG', 'en-orig',
        'hi', 'hi-IN', 'hi-Latn', 'hi-Deva'
        ]
                
        # Method 1: Try to get transcript with our preferred languages
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_data = ytt_api.fetch(
                video_id, 
                languages=language_preferences
            )
            print(f"Found transcript in preferred language")
            
            # Process the transcript data
            text_segments = []
            for segment in transcript_data:
                # Handle both dict and object formats
                if isinstance(segment, dict):
                    text_segments.append(segment.get('text', ''))
                else:
                    # For object format, access text attribute
                    text_segments.append(getattr(segment, 'text', ''))
            
            full_text = ' '.join(text_segments)
            return clean_transcript_text(full_text)
            
        except Exception as e:
            print(f"Could not get preferred language transcript: {e}")
            
            # Method 2: List all available transcripts and pick the best one
            try:
                # Get list of all available transcripts
                transcript_list = ytt_api.list(video_id)
                
                # Convert to list to iterate
                available_transcripts = []
                for transcript in transcript_list:
                    available_transcripts.append({
                        'language': transcript.language,
                        'language_code': transcript.language_code,
                        'is_generated': transcript.is_generated,
                        'is_translatable': transcript.is_translatable,
                        'transcript_obj': transcript
                    })
                
                if not available_transcripts:
                    print("No transcripts available")
                    return None
                
                # Sort by preference: English > Hindi > Others, Manual > Auto
                def sort_key(t):
                    lang_priority = 10  # Default low priority
                    if t['language_code'].startswith('en'):
                        lang_priority = 0
                    elif t['language_code'].startswith('hi'):
                        lang_priority = 1
                    
                    # Manual transcripts get +0, auto-generated get +5
                    auto_penalty = 5 if t['is_generated'] else 0
                    
                    return lang_priority + auto_penalty
                
                available_transcripts.sort(key=sort_key)
                best_transcript = available_transcripts[0]
                
                print(f"Selected: {best_transcript['language']} "
                        f"({'auto' if best_transcript['is_generated'] else 'manual'})")
                
                # Fetch the selected transcript
                transcript_data = best_transcript['transcript_obj'].fetch()
                
                # Process the transcript data
                text_segments = []
                for segment in transcript_data:
                    if isinstance(segment, dict):
                        text_segments.append(segment.get('text', ''))
                    else:
                        text_segments.append(getattr(segment, 'text', ''))
                
                full_text = ' '.join(text_segments)
                
                # If it's Hindi and translatable, try to translate to English
                if (best_transcript['language_code'].startswith('hi') and 
                    best_transcript['is_translatable']):
                    try:
                        print("Attempting to translate Hindi to English...")
                        translated = best_transcript['transcript_obj'].translate('en')
                        translated_data = translated.fetch()
                        
                        text_segments = []
                        for segment in translated_data:
                            if isinstance(segment, dict):
                                text_segments.append(segment.get('text', ''))
                            else:
                                text_segments.append(getattr(segment, 'text', ''))
                        
                        full_text = ' '.join(text_segments)
                        print("Successfully translated to English")
                    except Exception as e:
                        print(f"Translation failed, using original: {e}")
                
                return clean_transcript_text(full_text)
                
            except Exception as list_error:
                print(f"Error listing transcripts: {list_error}")
                
                # Method 3: Final fallback - try to get any transcript
                try:
                    ytt_api = YouTubeTranscriptApi()
                    transcript_data = ytt_api.fetch(video_id)
                    
                    text_segments = []
                    for segment in transcript_data:
                        if isinstance(segment, dict):
                            text_segments.append(segment.get('text', ''))
                        else:
                            text_segments.append(getattr(segment, 'text', ''))
                    
                    full_text = ' '.join(text_segments)
                    return clean_transcript_text(full_text)
                    
                except Exception as final_error:
                    print(f"Final fallback failed: {final_error}")
                    return None
                    
        except TranscriptsDisabled:
            print(f"Transcripts are disabled for video {video_id}")
            return None
        except Exception as e:
            if '429' in str(e):
                print(f"Rate limited by YouTube API: {e}")
                # Add extra delay for rate limiting
                time.sleep(5)
            else:
                print(f"Error with YouTube Transcript API: {e}")
            return None
        
    # def _clean_transcript_text(self, text):
    #     """Clean transcript text by removing duplicates and formatting issues."""
    #     # Remove excess whitespace
    #     text = re.sub(r'\s+', ' ', text)
        
    #     # Remove music/sound effect annotations
    #     text = re.sub(r'\[.*?\]', '', text)
    #     text = re.sub(r'\(.*?\)', '', text)
        
    #     return text.strip()
    