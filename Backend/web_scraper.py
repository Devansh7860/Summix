"""
Web scraper module for fetching transcripts from online services.
No AI/LLM dependencies - uses pure Playwright for web automation.
"""
import asyncio
import re
import sys
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor

try:
    from playwright.async_api import async_playwright, Browser, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️  Playwright not installed. Install with:")
    print("    pip install playwright")
    print("    playwright install chromium")


class WebTranscriptScraper:
    """Scrapes transcripts from various online services using browser automation."""

    def __init__(self):
        self.available = PLAYWRIGHT_AVAILABLE
        self._executor = ThreadPoolExecutor(max_workers=1)

    def fetch(self, video_url: str) -> Optional[str]:
        """
        Fetch transcript for a single video using web scraping.
        For playlists, use fetch_playlist() for better performance.
        """
        if not self.available:
            print("⚠️  Web scraping not available (missing dependencies)")
            return None
        
        print("Starting web scraping...")
        
        try:
            future = self._executor.submit(self._run_async_isolated, video_url)
            result = future.result(timeout=120)  # 120 second timeout
            return result
        except Exception as e:
            print(f"⚠️  Web scraping failed: {e}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return None

    def fetch_playlist(self, video_urls: List[str], max_concurrent: int = 3) -> Dict[str, Optional[str]]:
        """
        Efficiently fetch transcripts for multiple videos using concurrent processing.
        This processes videos in parallel batches for much better performance.
        
        Args:
            video_urls: List of YouTube video URLs
            max_concurrent: Maximum number of videos to process simultaneously (default: 3)
            
        Returns:
            Dictionary mapping video URLs to their transcripts (or None if failed)
        """
        if not self.available:
            print("⚠️  Web scraping not available (missing dependencies)")
            return {url: None for url in video_urls}
        
        print(f"Starting concurrent playlist scraping for {len(video_urls)} videos (max {max_concurrent} concurrent)...")
        
        try:
            future = self._executor.submit(self._run_playlist_async_isolated, video_urls, max_concurrent)
            result = future.result(timeout=600)  # 10 minute timeout for larger playlists
            return result
        except Exception as e:
            print(f"⚠️  Concurrent playlist approach failed: {e}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return {url: None for url in video_urls}

    def _run_async_isolated(self, video_url: str) -> Optional[str]:
        """Run async playwright in completely isolated event loop."""
        # Set Windows event loop policy if needed
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Create and run in completely new event loop
        return asyncio.run(self._fetch_single_video_async(video_url))
    
    def _run_playlist_async_isolated(self, video_urls: List[str], max_concurrent: int = 3) -> Dict[str, Optional[str]]:
        """Run playlist async in completely isolated event loop."""
        # Set Windows event loop policy if needed
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Create and run in completely new event loop
        return asyncio.run(self._fetch_playlist_async(video_urls, max_concurrent))

    async def _fetch_single_video_async(self, video_url: str) -> Optional[str]:
        """Fetch a single video transcript."""
        async with async_playwright() as p:
            launch_options = {
                'headless': True,
                'args': [
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-extensions',
                    '--no-first-run',
                    '--disable-default-apps',
                ]
            }
            
            # Add Windows-specific options
            import os
            if os.name == 'nt':
                launch_options['args'].extend([
                    '--disable-background-timer-throttling',
                    '--disable-renderer-backgrounding',
                    '--disable-backgrounding-occluded-windows',
                ])
            
            browser = await p.chromium.launch(**launch_options)
            
            try:
                print("Trying youtubetotranscript.com...")
                transcript = await self._scrape_youtubetotranscript(video_url, browser)
                if transcript and len(transcript) > 100:
                    print("SUCCESS: Successfully scraped transcript")
                    return transcript
                else:
                    print(f"⚠️  Returned empty/short transcript (length: {len(transcript) if transcript else 0})")
            except Exception as e:
                print(f"⚠️  Failed: {str(e)[:150]}")
            finally:
                await browser.close()
                
        return None

    async def _fetch_playlist_async(self, video_urls: List[str], max_concurrent: int = 3) -> Dict[str, Optional[str]]:
        """
        Efficiently fetch transcripts for multiple videos using concurrent processing.
        Uses semaphore to limit concurrent tabs and processes videos in parallel batches.
        """
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)
        
        print(f"Launching browser for {len(video_urls)} videos with max {max_concurrent} concurrent tabs...")
        
        async with async_playwright() as p:
            # Launch browser once for all videos
            launch_options = {
                'headless': True,
                'args': [
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-extensions',
                    '--no-first-run',
                    '--disable-default-apps',
                ]
            }
            
            # Add Windows-specific options
            import os
            if os.name == 'nt':
                launch_options['args'].extend([
                    '--disable-background-timer-throttling',
                    '--disable-renderer-backgrounding',
                    '--disable-backgrounding-occluded-windows',
                ])
            
            browser = await p.chromium.launch(**launch_options)
            
            async def process_single_video(video_url: str, index: int) -> tuple:
                """Process a single video with semaphore control and retry logic."""
                async with semaphore:
                    print(f"[{index+1}/{len(video_urls)}] Processing: {video_url}")
                    
                    # Add a small staggered delay to avoid overwhelming the server
                    await asyncio.sleep(index * 0.5)  # Stagger requests by 0.5s each
                    
                    # Try scraping service for this video with retry
                    transcript = None
                    max_retries = 2
                    
                    for attempt in range(max_retries + 1):
                        try:
                            if attempt > 0:
                                print(f"  [{index+1}] Retry attempt {attempt}/{max_retries}")
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s
                            
                            print(f"  [{index+1}] Trying youtubetotranscript.com...")
                            transcript = await self._scrape_youtubetotranscript(video_url, browser)
                            
                            if transcript and len(transcript) > 100:
                                print(f"  [{index+1}] SUCCESS! ({len(transcript)} chars)")
                                break
                            else:
                                print(f"  [{index+1}] ⚠️  Returned short/empty transcript (attempt {attempt+1})")
                                if attempt < max_retries:
                                    await asyncio.sleep(1)  # Short delay before retry
                                    
                        except Exception as e:
                            print(f"  [{index+1}] ❌ Attempt {attempt+1} failed: {str(e)[:100]}")
                            if attempt < max_retries:
                                await asyncio.sleep(2)  # Wait before retry
                            else:
                                # Final attempt failed
                                break
                    
                    if transcript:
                        print(f"[{index+1}] ✅ Completed successfully")
                    else:
                        print(f"[{index+1}] ❌ Failed after {max_retries + 1} attempts")
                    
                    # Small delay after completion to be respectful
                    await asyncio.sleep(0.5)
                    
                    return video_url, transcript
            
            try:
                # Create tasks for all videos
                tasks = [
                    process_single_video(video_url, i) 
                    for i, video_url in enumerate(video_urls)
                ]
                
                # Process all videos concurrently with semaphore limiting
                print(f"Starting concurrent processing of {len(tasks)} videos...")
                completed_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results
                for i, result in enumerate(completed_results):
                    if isinstance(result, Exception):
                        print(f"⚠️  Task {i+1} failed with exception: {result}")
                        # Add failed video to results
                        if i < len(video_urls):
                            results[video_urls[i]] = None
                        continue
                    video_url, transcript = result
                    results[video_url] = transcript
                        
            finally:
                await browser.close()
                success_count = sum(1 for r in results.values() if r)
                print(f"🎉 Concurrent processing complete! Success: {success_count}/{len(video_urls)}")
        
        return results

    async def _scrape_youtubetotranscript(self, video_url: str, browser) -> Optional[str]:
        """Scrape transcript from youtubetotranscript.com with improved error handling."""
        # Create isolated context for this video to avoid conflicts
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        await context.grant_permissions(['clipboard-read'])
        page = await context.new_page()
        
        try:
            # Add extra timeout for better reliability
            await page.goto("https://youtubetotranscript.com/", timeout=45000)
            await self._handle_cookie_banner_async(page)

            # Wait for page to be fully loaded
            await page.wait_for_load_state('networkidle', timeout=10000)

            input_selector = 'input[type="text"], input[name="youtube_url"]'
            await page.wait_for_selector(input_selector, timeout=15000)
            
            # Clear input field first and then fill
            await page.fill(input_selector, "")
            await page.fill(input_selector, video_url)
            
            # Wait a bit for the input to be processed
            await page.wait_for_timeout(500)
            
            await page.click('button[type="submit"]')
            
            # Wait for the transcript to be generated
            copy_button_selector = '#copy-transcript'
            await page.wait_for_selector(copy_button_selector, timeout=30000)
            
            # Scroll to the copy button to ensure it's visible
            await page.locator(copy_button_selector).scroll_into_view_if_needed()
            
            # Click the copy button
            await page.click(copy_button_selector)
            
            # Wait longer for clipboard operation to complete
            await page.wait_for_timeout(1500)
            
            # Try to read from clipboard with retry
            transcript = None
            for clip_attempt in range(3):
                try:
                    transcript = await page.evaluate("navigator.clipboard.readText()")
                    if transcript and len(transcript) > 50:
                        break
                    await page.wait_for_timeout(1000)  # Wait 1s between clipboard attempts
                except Exception as clip_error:
                    if clip_attempt < 2:
                        await page.wait_for_timeout(1000)
                        continue
                    else:
                        raise clip_error
            
            if transcript and len(transcript) > 50:
                return self._clean_text(transcript)
                
        except Exception as e:
            # Log more detailed error info for debugging
            error_msg = str(e)[:200]
            print(f"    Detailed error in scraper: {error_msg}")
            raise e
        finally:
            await context.close()
            
        return None

    async def _handle_cookie_banner_async(self, page):
        """Handle cookie consent banners."""
        try:
            cookie_selectors = [
                'button:has-text("Accept")',
                'button:has-text("Allow")',
                'button:has-text("OK")',
                '[data-testid="accept-cookies"]',
                '.cookie-accept',
                '#accept-cookies'
            ]
            
            for selector in cookie_selectors:
                try:
                    await page.click(selector, timeout=2000)
                    await page.wait_for_timeout(500)
                    break
                except:
                    continue
        except:
            pass

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        text = re.sub(r'^\d{1,2}:\d{2}\s', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        return text.strip()

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Test block
if __name__ == "__main__":
    # Test single video
    TEST_URL = "https://www.youtube.com/watch?v=GOejI6c0CMQ"
    print(f"=== Single Video Test ===")
    print(f"Fetching transcript for: {TEST_URL}")
    
    scraper = WebTranscriptScraper()
    transcript = scraper.fetch(TEST_URL)
    
    if transcript:
        print(f"\n✅ Single video SUCCESS")
        print(f"Transcript length: {len(transcript)} chars")
        print("Preview:", transcript[:100] + "...")
    else:
        print("\n❌ Single video FAILED")
    
    # Test concurrent playlist processing  
    print(f"\n=== Concurrent Playlist Test ===")
    playlist_urls = [
        "https://www.youtube.com/watch?v=Q57_iaGrxLg&list=PL0vfts4VzfNjS8-DG68UDBZyeVkND3AXt&index=1&pp=iAQB",  
        "https://www.youtube.com/watch?v=ozkg_iW9mNU&list=PL0vfts4VzfNjS8-DG68UDBZyeVkND3AXt&index=2&pp=iAQB",  
        "https://www.youtube.com/watch?v=7rXgVsIGvGQ&list=PL0vfts4VzfNjS8-DG68UDBZyeVkND3AXt&index=3&pp=iAQB",
        "https://www.youtube.com/watch?v=ek2yOqAIYuU&list=PL0vfts4VzfNjS8-DG68UDBZyeVkND3AXt&index=4&pp=iAQB",
        "https://www.youtube.com/watch?v=TsKHjFeonRE&list=PL0vfts4VzfNjS8-DG68UDBZyeVkND3AXt&index=5&pp=iAQB",
        "https://www.youtube.com/watch?v=wbQwD3QS19I&list=PL0vfts4VzfNjS8-DG68UDBZyeVkND3AXt&index=6&pp=iAQB0gcJCesJAYcqIYzv"  
    ]
    
    print(f"Testing concurrent processing with {len(playlist_urls)} videos...")
    playlist_results = scraper.fetch_playlist(playlist_urls, max_concurrent=3)
    
    print(f"\n=== Playlist Results ===")
    for i, (url, transcript) in enumerate(playlist_results.items(), 1):
        status = "✅ SUCCESS" if transcript else "❌ FAILED"
        length = len(transcript) if transcript else 0
        print(f"Video {i}: {status} ({length} chars)")
    
    success_rate = sum(1 for t in playlist_results.values() if t) / len(playlist_urls) * 100
    print(f"\nOverall success rate: {success_rate:.1f}%")