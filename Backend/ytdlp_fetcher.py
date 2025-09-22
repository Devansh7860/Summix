"""
yt-dlp based transcript fetcher module.
"""
import os
import glob
import time
import re
from typing import Optional

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    print("⚠️  yt-dlp not installed. Install with: pip install yt-dlp")


class YtDlpTranscriptFetcher:
    """Fetches transcripts using yt-dlp."""
    
    def __init__(self):
        self.available = YTDLP_AVAILABLE
    
    def fetch(self, video_url: str, video_id: str) -> Optional[str]:
        """
        Fetch transcript using yt-dlp.
        
        Args:
            video_url: YouTube video URL
            video_id: YouTube video ID
            
        Returns:
            Transcript text or None
        """
        if not self.available:
            return None
        
        # try:
        #     ydl_opts = {
        #         'writesubtitles': True,
        #         'writeautomaticsub': True,
        #         'subtitleslangs': ['en', 'en-US', 'en-GB', 'hi'],
        #         'skip_download': True,
        #         'outtmpl': f'{video_id}.%(lang)s.%(ext)s',
        #         'quiet': True,
        #         'no_warnings': True,
        #         'socket_timeout': 30,
        #         'retries': 2,
        #     }
            
        #     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        #         ydl.download([video_url])
            
        #     # Find downloaded files
        #     downloaded_files = glob.glob(f"{video_id}.*.[vst][rst][bt]")
            
        #     if not downloaded_files:
        #         print("⚠️  No subtitle files downloaded")
        #         return None
            
        #     # Read and process the file
        #     with open(downloaded_files[0], 'r', encoding='utf-8') as f:
        #         content = f.read()
            
        #     # Clean up files
        #     self._cleanup_files(video_id)
            
        #     return self._process_subtitle_content(content)
            
        # except Exception as e:
        #     if '429' in str(e):
        #         print(f"⚠️  Rate limited (429)")
        #         time.sleep(10)
        #     else:
        #         print(f"⚠️  Error with yt-dlp: {e}")
            
        #     self._cleanup_files(video_id)
        #     return None


        """
        Fallback method using yt-dlp with optimized settings.
        """
        try:
            # First, get subtitle info without downloading
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                # Check available subtitles
                manual_subs = info.get('subtitles', {})
                auto_subs = info.get('automatic_captions', {})
                
                # Determine best subtitle to download
                target_lang = None
                use_auto = False
                
                # Priority order
                if 'en' in manual_subs or any(k.startswith('en') for k in manual_subs):
                    target_lang = 'en' if 'en' in manual_subs else next(k for k in manual_subs if k.startswith('en'))
                    use_auto = False
                elif 'en' in auto_subs or any(k.startswith('en') for k in auto_subs):
                    target_lang = 'en' if 'en' in auto_subs else next(k for k in auto_subs if k.startswith('en'))
                    use_auto = True
                elif 'hi' in manual_subs:
                    target_lang = 'hi'
                    use_auto = False
                elif 'hi' in auto_subs:
                    target_lang = 'hi'
                    use_auto = True
                elif manual_subs:
                    target_lang = list(manual_subs.keys())[0]
                    use_auto = False
                elif auto_subs:
                    target_lang = list(auto_subs.keys())[0]
                    use_auto = True
                else:
                    return None
                
                print(f"Downloading {target_lang} subtitle ({'auto' if use_auto else 'manual'})")
                
                # Download the selected subtitle
                download_opts = {
                    'writesubtitles': not use_auto,
                    'writeautomaticsub': use_auto,
                    'subtitleslangs': [target_lang],
                    'skip_download': True,
                    'outtmpl': f'{video_id}.%(lang)s.%(ext)s',
                    'quiet': True,
                    'no_warnings': True,
                    # Add network optimizations
                    'socket_timeout': 30,
                    'retries': 2,
                    'fragment_retries': 2,
                    # Add headers to appear more like a browser
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept-Language': 'en-US,en;q=0.9',
                    }
                }
                
                with yt_dlp.YoutubeDL(download_opts) as ydl_download:
                    ydl_download.download([video_url])
                
                # Find and process the downloaded file
                downloaded_files = glob.glob(f"{video_id}.*.vtt")
                
                if not downloaded_files:
                    return None
                
                with open(downloaded_files[0], 'r', encoding='utf-8') as f:
                    vtt_content = f.read()
                
                # Clean up files
                self._cleanup_files(video_id)
                
                return self._process_subtitle_content(vtt_content)
                
        except Exception as e:
            if '429' in str(e):
                print(f"Rate limited (429): {e}")
                time.sleep(10)  # Longer wait for yt-dlp rate limiting
            else:
                print(f"Error with yt-dlp: {e}")
            
            self._cleanup_files(video_id)
            return None
    
    def _process_subtitle_content(self, content: str) -> str:
        """Process VTT/SRT content to extract clean text."""
        lines = []
        for line in content.splitlines():
            # Skip timestamps and headers
            if ('-->' in line or 
                line.strip().isdigit() or 
                'WEBVTT' in line or
                re.match(r'^\d{2}:\d{2}', line)):
                continue
            
            # Remove HTML tags
            cleaned = re.sub(r'<[^>]+>', '', line).strip()
            
            # Skip annotations
            if cleaned and not cleaned.startswith('[') and not cleaned.startswith('('):
                lines.append(cleaned)
        
        # De-duplicate consecutive lines
        if not lines:
            return ""
        
        final_lines = [lines[0]]
        for i in range(1, len(lines)):
            if lines[i] != final_lines[-1]:
                final_lines.append(lines[i])
        
        return ' '.join(final_lines)
    
    def _cleanup_files(self, video_id: str):
        """Clean up downloaded files."""
        patterns = [f"{video_id}.*"]
        for pattern in patterns:
            for file_path in glob.glob(pattern):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except:
                    pass