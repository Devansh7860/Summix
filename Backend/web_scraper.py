"""
Web scraper module for fetching transcripts from online services.
Uses Browserless.io cloud API for fast performance and easy deployment.
"""
import re
import requests
import time
from typing import Optional, List, Dict

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("⚠️  BeautifulSoup not installed. Install with: pip install beautifulsoup4")

# Browserless.io API configuration
BROWSERLESS_API_KEY = "2T9IYdLEgrhmoZYa6510e1a6d38be8c1e8d6debd4012fb7e3"
BROWSERLESS_ENDPOINT = f"https://production-sfo.browserless.io/content?token={BROWSERLESS_API_KEY}"
BROWSERLESS_ENABLED = bool(BROWSERLESS_API_KEY and BROWSERLESS_API_KEY != "YOUR_API_KEY_HERE")

# Configuration constants
MIN_TRANSCRIPT_LENGTH = 100
MIN_CONTENT_CHECK_LENGTH = 30
REQUEST_TIMEOUT = 60
BROWSER_TIMEOUT = 30000
REQUEST_DELAY = 1.0


class WebTranscriptScraper:
    """
    Fast transcript scraper using Browserless.io cloud API.
    No browser installation required - uses remote cloud browsers.
    """

    def __init__(self, browserless_api_key: Optional[str] = None):
        # Use provided API key or fallback to hardcoded one
        self.browserless_api_key = browserless_api_key or BROWSERLESS_API_KEY
        
        if not self.browserless_api_key or self.browserless_api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Browserless API key is required")
        if not BEAUTIFULSOUP_AVAILABLE:
            raise ImportError("BeautifulSoup is required. Install with: pip install beautifulsoup4")
            
        # Create endpoint with the API key
        self.browserless_endpoint = f"https://production-sfo.browserless.io/content?token={self.browserless_api_key}"
        
        self.available = True
        self._browserless_session = requests.Session()
        self._browserless_session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        # Performance tracking
        self._stats = {
            'attempts': 0,
            'successes': 0,
            'total_time': 0.0
        }

    def fetch(self, video_url: str) -> Optional[str]:
        """
        Fetch transcript for a single video using Browserless.io cloud API
        """
        if not video_url or not isinstance(video_url, str):
            print("❌ Invalid video URL provided")
            return None
            
        if not self.available:
            print("⚠️  Web scraping not available")
            return None
        
        print("🚀 Fetching transcript via Browserless.io...")
        self._stats['attempts'] += 1
        start_time = time.time()
        
        transcript = self._fetch_via_browserless(video_url)
        duration = time.time() - start_time
        self._stats['total_time'] += duration
        
        if transcript:
            self._stats['successes'] += 1
            print(f"✅ Success! ({len(transcript)} chars in {duration:.1f}s)")
            self._log_performance_stats()
            return transcript
        else:
            print(f"❌ Failed after {duration:.1f}s")
            self._log_performance_stats()
            return None

    def _fetch_via_browserless(self, video_url: str) -> Optional[str]:
        """
        Fast transcript fetching using Browserless.io cloud API
        """
        try:
            # Extract video ID from YouTube URL
            video_id = self._extract_video_id(video_url)
            if not video_id:
                print("❌ Could not extract video ID from URL")
                return None
            
            # Navigate directly to transcript page
            transcript_url = f"https://youtubetotranscript.com/transcript?v={video_id}"
            
            payload = {
                "url": transcript_url,
                "gotoOptions": {
                    "timeout": BROWSER_TIMEOUT,
                    "waitUntil": "networkidle2"
                }
            }
            
            start_time = time.time()
            response = self._browserless_session.post(
                self.browserless_endpoint,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                content = response.text
                transcript = self._extract_transcript_from_html(content)
                
                if transcript and len(transcript) > MIN_TRANSCRIPT_LENGTH:
                    print(f"⚡ Browserless completed in {duration:.1f}s")
                    return transcript
                else:
                    print(f"⚠️  No meaningful transcript found (duration: {duration:.1f}s)")
                    return None
            else:
                print(f"❌ Browserless API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Browserless error: {e}")
            return None

    def _extract_video_id(self, youtube_url: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats"""
        # Single comprehensive pattern that handles all cases
        pattern = r'(?:youtube\.com/(?:watch\?.*?v=|embed/)|youtu\.be/)([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, youtube_url)
        return match.group(1) if match else None

    def _extract_transcript_from_html(self, html_content: str) -> Optional[str]:
        """Extract transcript from HTML using BeautifulSoup"""
        if not html_content or len(html_content) < 100:
            return None
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            candidates = []
            
            # Strategy 1: Look for textareas (common for transcript display)
            for textarea in soup.find_all('textarea'):
                text = textarea.get_text(strip=True) or textarea.get('value', '')
                if text and len(text) > MIN_TRANSCRIPT_LENGTH:
                    candidates.append(text)
            
            # Strategy 2: Look for pre elements
            for pre in soup.find_all('pre'):
                text = pre.get_text(strip=True)
                if text and len(text) > MIN_TRANSCRIPT_LENGTH:
                    candidates.append(text)
            
            # Strategy 3: Content analysis for substantial text blocks
            for element in soup.find_all(['div', 'p', 'span']):
                text = element.get_text(strip=True)
                if len(text) > 200:
                    candidates.append(text)
            
            # Process candidates and return the best one
            for candidate in sorted(candidates, key=len, reverse=True):
                cleaned = self._clean_transcript_text(candidate)
                if self._is_real_transcript_content(cleaned):
                    return cleaned
            
            return None
            
        except Exception as e:
            print(f"❌ HTML parsing error: {e}")
            return None

    def _is_real_transcript_content(self, text: str) -> bool:
        """Check if text looks like actual video transcript content"""
        if not text or len(text) < MIN_CONTENT_CHECK_LENGTH:
            return False
        
        text_lower = text.lower()
        
        # Red flags - UI/promotional text
        ui_keywords = [
            'youtube to transcript', 'free transcript', 'bookmark us',
            'generate youtube', 'easy copy and edit', 'supports translation',
            'multiple languages', 'ctrl+d', 'cmd+d', 'paste youtube url'
        ]
        
        ui_count = sum(1 for keyword in ui_keywords if keyword in text_lower)
        if ui_count > 1:
            return False
        
        # Good indicators - actual speech content
        word_count = len(text.split())
        sentence_indicators = text.count('.') + text.count('!') + text.count('?')
        
        return word_count > 15 and (sentence_indicators > 2 or word_count > 50)

    def _clean_transcript_text(self, text: str) -> str:
        """Clean transcript text while preserving actual content"""
        if not text:
            return ""
        
        # Remove timestamps
        text = re.sub(r'\b\d{1,2}:\d{2}\b', '', text)
        
        # Remove brackets and parentheses
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        
        # Remove UI-specific patterns
        ui_patterns = [
            r'youtube\s+to\s+transcript.*?free',
            r'bookmark\s+us.*?ctrl\+d',
            r'generate\s+youtube\s+transcript.*?languages'
        ]
        
        for pattern in ui_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def fetch_playlist(self, video_urls: List[str], max_concurrent: int = 3) -> Dict[str, Optional[str]]:
        """
        Fetch transcripts for multiple videos sequentially using Browserless.io.
        
        Args:
            video_urls: List of YouTube video URLs
            max_concurrent: Not used in this implementation, kept for compatibility
            
        Returns:
            Dictionary mapping video URLs to their transcripts (or None if failed)
        """
        if not video_urls or not isinstance(video_urls, list):
            print("❌ Invalid video URLs list provided")
            return {}
            
        if not self.available:
            print("⚠️  Web scraping not available")
            return {url: None for url in video_urls}
        
        print(f"🚀 Processing {len(video_urls)} videos sequentially...")
        results = {}
        
        for i, video_url in enumerate(video_urls, 1):
            print(f"\n[{i}/{len(video_urls)}] Processing: {video_url}")
            transcript = self.fetch(video_url)
            results[video_url] = transcript
            
            # Small delay between requests to be respectful
            if i < len(video_urls):
                time.sleep(REQUEST_DELAY)
        
        success_count = sum(1 for t in results.values() if t)
        print(f"\n🎉 Playlist complete! Success: {success_count}/{len(video_urls)}")
        return results

    def _log_performance_stats(self):
        """Log performance statistics for monitoring"""
        if self._stats['attempts'] > 0:
            success_rate = (self._stats['successes'] / self._stats['attempts']) * 100
            avg_time = self._stats['total_time'] / self._stats['attempts']
            
            print(f"📊 Performance: {self._stats['successes']}/{self._stats['attempts']} successful ({success_rate:.1f}%), "
                  f"avg {avg_time:.1f}s per request")


# Test block
if __name__ == "__main__":
    # Test single video
    TEST_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print(f"=== Browserless.io Cloud API Test ===")
    print(f"Fetching transcript for: {TEST_URL}")
    
    scraper = WebTranscriptScraper()
    transcript = scraper.fetch(TEST_URL)
    
    if transcript:
        print(f"\n✅ SUCCESS")
        print(f"Transcript length: {len(transcript)} chars")
        print("Preview:", transcript[:200] + "...")
    else:
        print("\n❌ FAILED")
    
    # Test playlist
    print(f"\n=== Playlist Test ===")
    playlist_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll
        "https://youtu.be/jNQXAC9IVRw",  # Me at the zoo
        "https://www.youtube.com/watch?v=9bZkp7q19f0",  # Gangnam Style
    ]
    
    print(f"Testing {len(playlist_urls)} videos...")
    playlist_results = scraper.fetch_playlist(playlist_urls, max_concurrent=2)
    
    print(f"\n=== Results ===")
    for i, (url, transcript) in enumerate(playlist_results.items(), 1):
        status = "✅ SUCCESS" if transcript else "❌ FAILED"
        length = len(transcript) if transcript else 0
        print(f"Video {i}: {status} ({length} chars)")
    
    success_rate = sum(1 for t in playlist_results.values() if t) / len(playlist_urls) * 100
    print(f"\nOverall success rate: {success_rate:.1f}%")