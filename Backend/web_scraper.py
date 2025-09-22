
# """
# Web scraper module for fetching transcripts from online services.
# No AI/LLM dependencies - uses pure Playwright for web automation.
# """
# import asyncio
# import re
# from typing import Optional

# try:
#     from playwright.async_api import async_playwright, Browser, TimeoutError as PlaywrightTimeoutError
#     import nest_asyncio
#     nest_asyncio.apply()  # Allow nested event loops
#     PLAYWRIGHT_AVAILABLE = True
# except ImportError:
#     PLAYWRIGHT_AVAILABLE = False
#     print("⚠️  Playwright not installed. Install with:")
#     print("    pip install playwright nest-asyncio")
#     print("    playwright install chromium")


# class WebTranscriptScraper:
#     """Scrapes transcripts from various online services using browser automation."""

#     def __init__(self):
#         self.available = PLAYWRIGHT_AVAILABLE

#     def fetch(self, video_url: str) -> Optional[str]:
#         """
#         Fetch transcript using web scraping.
#         """
#         if not self.available:
#             print("⚠️  Web scraping not available (missing dependencies)")
#             return None
#         try:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             result = loop.run_until_complete(self._fetch_async(video_url))
#             loop.close()
#             return result
#         except Exception as e:
#             print(f"⚠️  Error in web scraping: {e}")
#             return None

#     async def _fetch_async(self, video_url: str) -> Optional[str]:
#         """
#         Launches one browser instance and tries multiple scrapers in separate tabs.
#         This is much faster than launching a new browser for each site.
#         """
#         # Updated list of scrapers, with downsub.com replaced by kome.ai
#         scrapers = [
#             ("youtubetotranscript.com", self._scrape_youtubetotranscript),
#             ("tactiq.io", self._scrape_tactiq),
#             ("kome.ai", self._scrape_komeai),
#         ]

#         async with async_playwright() as p:
#             browser = await p.chromium.launch(headless=True)
#             for service_name, scraper_func in scrapers:
#                 try:
#                     print(f"🌐 Trying {service_name}...")
#                     transcript = await scraper_func(video_url, browser)
#                     if transcript and len(transcript) > 100:
#                         print(f"✅ Successfully scraped from {service_name}")
#                         await browser.close()
#                         return transcript
#                 except Exception as e:
#                     print(f"⚠️  Failed with {service_name}: {str(e)[:150]}")
#                     continue
#             await browser.close()
#         return None

#     async def _handle_cookie_banner(self, page):
#         """Pre-emptively clicks common cookie consent buttons."""
#         consent_buttons = [
#             page.locator('button:has-text("Accept")'),
#             page.locator('button:has-text("Accept all")'),
#             page.locator('button:has-text("I agree")'),
#             page.locator('button:has-text("Allow all")'),
#         ]
#         try:
#             # Click the first visible button, waiting max 3 seconds
#             for button in consent_buttons:
#                 if await button.is_visible(timeout=3000):
#                     await button.click()
#                     print(" ✓ Handled cookie banner.")
#                     return
#         except PlaywrightTimeoutError:
#             # No banner found or it's not clickable, which is fine.
#             pass

#     async def _scrape_youtubetotranscript(self, video_url: str, browser: Browser) -> Optional[str]:
#         """Scrapes from youtubetotranscript.com using its copy button. (UNCHANGED)"""
#         context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
#         await context.grant_permissions(['clipboard-read'])
#         page = await context.new_page()
#         try:
#             await page.goto("https://youtubetotranscript.com/", timeout=30000)
#             await self._handle_cookie_banner(page)

#             input_selector = 'input[type="text"], input[name="youtube_url"]'
#             await page.wait_for_selector(input_selector, timeout=10000)
#             await page.fill(input_selector, video_url)
            
#             await page.click('button[type="submit"]')
            
#             copy_button_selector = '#copy-transcript'
#             await page.wait_for_selector(copy_button_selector, timeout=20000)
#             await page.click(copy_button_selector)
            
#             transcript = await page.evaluate("navigator.clipboard.readText()")
#             if transcript and len(transcript) > 50:
#                 return self._clean_text(transcript)
#         finally:
#             await context.close()
#         return None

#     async def _scrape_tactiq(self, video_url: str, browser: Browser) -> Optional[str]:
#         """Scrapes from tactiq.io using its copy button. (FIXED)"""
#         context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
#         await context.grant_permissions(['clipboard-read'])
#         page = await context.new_page()
#         try:
#             await page.goto("https://tactiq.io/tools/youtube-transcript", timeout=30000)
#             await self._handle_cookie_banner(page)

#             await page.fill("input[name='yt'] , input[placeholder='Enter YouTube URL.. https://www.youtube.com/watch?v=Mcm3CDNMnd0']", video_url)

#             await page.click("input[type='submit']")

#             await page.wait_for_selector("div#transcript", timeout=20000)

#             copy_button_selector = 'a#copy'
#             await page.wait_for_selector(copy_button_selector, timeout=20000)
#             await page.click(copy_button_selector)

#             # FIX: Wait a moment for the clipboard JavaScript to execute
#             await page.wait_for_timeout(500)

#             transcript = await page.evaluate("navigator.clipboard.readText()")
#             if transcript and len(transcript) > 50:
#                 return self._clean_text(transcript)

#         finally:
#             await context.close()
#         return None

#     async def _scrape_komeai(self, video_url: str, browser: Browser) -> Optional[str]:
#         """Scrapes from kome.ai by using its copy button. (NEW)"""
#         context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
#         await context.grant_permissions(['clipboard-read'])
#         page = await context.new_page()
#         try:
#             await page.goto("https://kome.ai/tools/youtube-transcript-generator", timeout=30000)
#             await self._handle_cookie_banner(page)

#             await page.fill('input[name="url"]', video_url)
#             await page.click('button[type="submit"]')

#             await page.wait_for_selector("div.form_transcript__gnYky", timeout=20000)

#             # Wait for the specific copy button class
#             copy_button_selector = 'button.form_copyButton__KsbsH'
#             await page.wait_for_selector(copy_button_selector, timeout=20000)
#             await page.click(copy_button_selector)
            
#             # Wait a moment for the clipboard JavaScript to execute
#             await page.wait_for_timeout(500)

#             transcript = await page.evaluate("navigator.clipboard.readText()")
#             if transcript and len(transcript) > 50:
#                 return self._clean_text(transcript)
#         finally:
#             await context.close()
#         return None

#     def _clean_text(self, text: str) -> str:
#         """Basic text cleaning."""
#         text = re.sub(r'^\d{1,2}:\d{2}\s', '', text, flags=re.MULTILINE)
#         text = re.sub(r'\s+', ' ', text)
#         text = re.sub(r'\[.*?\]', '', text)
#         text = re.sub(r'\(.*?\)', '', text)
#         return text.strip()

#     def _clean_srt_format(self, text: str) -> str:
#         """Clean SRT/VTT formatted text."""
#         lines = []
#         for line in text.splitlines():
#             if '-->' in line or line.strip().isdigit() or re.match(r'^\d{2}:\d{2}', line) or 'WEBVTT' in line:
#                 continue
#             cleaned = line.strip()
#             if cleaned and not cleaned.startswith('[') and not cleaned.startswith('('):
#                 lines.append(cleaned)

#         seen = set()
#         unique_lines = [x for x in lines if not (x in seen or seen.add(x))]
#         return ' '.join(unique_lines)

# # This block allows the script to be run directly for testing
# if __name__ == "__main__":
#     TEST_URL = "https://www.youtube.com/watch?v=pOF11EDprxc" # A different video to test
#     print(f"--- Running Test ---")
#     print(f"Fetching transcript for: {TEST_URL}")
    
#     scraper = WebTranscriptScraper()
#     transcript = scraper.fetch(TEST_URL)
    
#     if transcript:
#         print("\n--- ✅ SUCCESS ---")
#         print("Transcript (first 300 chars):")
#         print(transcript[:300] + "...")
#     else:
#         print("\n--- ❌ FAILED ---")
#         print("Could not retrieve transcript.")





"""
Web scraper module for fetching transcripts from online services.
No AI/LLM dependencies - uses pure Playwright for web automation.
"""
import asyncio
import re
import sys
import threading
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor

try:
    from playwright.async_api import async_playwright, Browser, TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
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
        
        # Always use the threaded async approach to avoid any sync/async conflicts
        # This is the most reliable approach for server environments
        try:
            print("Using threaded async approach for maximum reliability...")
            return self._fetch_sync_in_thread(video_url)
        except Exception as e:
            print(f"⚠️  Threaded async approach failed: {e}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return None

    def fetch_playlist(self, video_urls: List[str]) -> Dict[str, Optional[str]]:
        """
        Efficiently fetch transcripts for multiple videos using a single browser instance.
        This is much more efficient than calling fetch() multiple times.
        
        Args:
            video_urls: List of YouTube video URLs
            
        Returns:
            Dictionary mapping video URLs to their transcripts (or None if failed)
        """
        if not self.available:
            print("⚠️  Web scraping not available (missing dependencies)")
            return {url: None for url in video_urls}
        
        print(f"Starting playlist web scraping for {len(video_urls)} videos...")
        
        try:
            print("Using optimized playlist approach with shared browser...")
            future = self._executor.submit(self._run_playlist_async_isolated, video_urls)
            result = future.result(timeout=300)  # 5 minute timeout for playlists
            return result
        except Exception as e:
            print(f"⚠️  Playlist approach failed: {e}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return {url: None for url in video_urls}
        except Exception as e:
            print(f"⚠️  All web scraping approaches failed: {e}")
            return None

    def _fetch_sync_in_thread(self, video_url: str) -> Optional[str]:
        """Run async version in a separate thread with isolated event loop."""
        try:
            future = self._executor.submit(self._run_async_isolated, video_url)
            result = future.result(timeout=60)  # 60 second timeout
            return result
        except Exception as e:
            print(f"⚠️  Threaded approach failed: {str(e)}")
            import traceback
            print(f"🔍 Thread error details: {traceback.format_exc()}")
            return None
    
    def _run_async_isolated(self, video_url: str) -> Optional[str]:
        """Run async playwright in completely isolated event loop."""
        import asyncio
        import sys
        
        # Set Windows event loop policy if needed
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Create and run in completely new event loop
        return asyncio.run(self._fetch_async_isolated(video_url))
    
    def _run_playlist_async_isolated(self, video_urls: List[str]) -> Dict[str, Optional[str]]:
        """Run playlist async in completely isolated event loop."""
        import asyncio
        import sys
        
        # Set Windows event loop policy if needed
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Create and run in completely new event loop
        return asyncio.run(self._fetch_playlist_async_isolated(video_urls))
    
    async def _fetch_async_isolated(self, video_url: str) -> Optional[str]:
        """Async version with proper isolation."""
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            # Launch with Windows-friendly options
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
                for service_name, scraper_func in [
                    ("youtubetotranscript.com", self._scrape_youtubetotranscript_async),
                    ("tactiq.io", self._scrape_tactiq_async),
                    ("kome.ai", self._scrape_komeai_async),
                ]:
                    try:
                        print(f"Trying {service_name}...")
                        transcript = await scraper_func(video_url, browser)
                        if transcript and len(transcript) > 100:
                            print(f"SUCCESS: Successfully scraped from {service_name}")
                            return transcript
                        else:
                            print(f"⚠️  {service_name} returned empty/short transcript (length: {len(transcript) if transcript else 0})")
                    except Exception as e:
                        print(f"⚠️  Failed with {service_name}: {str(e)[:150]}")
                        import traceback
                        print(f"🔍 {service_name} error details: {traceback.format_exc()[:300]}")
                        continue
            finally:
                await browser.close()
                
        return None

    async def _fetch_playlist_async_isolated(self, video_urls: List[str]) -> Dict[str, Optional[str]]:
        """
        Efficiently fetch transcripts for multiple videos using a single browser.
        This approach reuses the same browser instance across all videos.
        """
        from playwright.async_api import async_playwright
        
        results = {}
        
        print(f"Launching single browser for {len(video_urls)} videos...")
        
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
            
            try:
                # Process each video in the same browser
                for i, video_url in enumerate(video_urls, 1):
                    print(f"Processing video {i}/{len(video_urls)}: {video_url}")
                    
                    # Try each scraping service for this video
                    transcript = None
                    for service_name, scraper_func in [
                        ("youtubetotranscript.com", self._scrape_youtubetotranscript_async),
                        ("tactiq.io", self._scrape_tactiq_async),
                        ("kome.ai", self._scrape_komeai_async),
                    ]:
                        try:
                            print(f"  Trying {service_name}...")
                            transcript = await scraper_func(video_url, browser)
                            if transcript and len(transcript) > 100:
                                print(f"  SUCCESS with {service_name}! ({len(transcript)} chars)")
                                break
                            else:
                                print(f"  ⚠️  {service_name} returned short/empty transcript")
                        except Exception as e:
                            print(f"  ❌ {service_name} failed: {str(e)[:100]}")
                            continue
                    
                    # Store result for this video
                    results[video_url] = transcript
                    
                    if transcript:
                        print(f"Video {i} completed successfully")
                    else:
                        print(f"Video {i} failed - no transcript available")
                    
                    # Small delay between videos to be respectful
                    if i < len(video_urls):
                        await asyncio.sleep(1)
                        
            finally:
                await browser.close()
                print(f"Playlist processing complete. Success: {sum(1 for r in results.values() if r)}/{len(video_urls)}")
        
        return results

    def _fetch_sync(self, video_url: str) -> Optional[str]:
        """Synchronous version using sync playwright with Windows fixes."""
        scrapers = [
            ("youtubetotranscript.com", self._scrape_youtubetotranscript_sync),
            ("tactiq.io", self._scrape_tactiq_sync),
            ("kome.ai", self._scrape_komeai_sync),
        ]

        try:
            # Critical fix for Windows subprocess issues in threaded context
            import os
            import asyncio
            if os.name == 'nt':  # Windows
                # Set the correct event loop policy for this thread
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                
                # Set subprocess creation flags to avoid issues
                import subprocess
                original_popen = subprocess.Popen
                def patched_popen(*args, **kwargs):
                    # Add Windows-specific flags to avoid subprocess errors
                    if 'creationflags' not in kwargs:
                        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                    return original_popen(*args, **kwargs)
                subprocess.Popen = patched_popen
            
            with sync_playwright() as p:
                # Launch with additional Windows-friendly options
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
                if os.name == 'nt':
                    launch_options['args'].extend([
                        '--disable-background-timer-throttling',
                        '--disable-renderer-backgrounding',
                        '--disable-backgrounding-occluded-windows',
                    ])
                
                browser = p.chromium.launch(**launch_options)
                try:
                    for service_name, scraper_func in scrapers:
                        try:
                            print(f"Trying {service_name}...")
                            transcript = scraper_func(video_url, browser)
                            if transcript and len(transcript) > 100:
                                print(f"SUCCESS: Successfully scraped from {service_name}")
                                return transcript
                            else:
                                print(f"⚠️  {service_name} returned empty/short transcript (length: {len(transcript) if transcript else 0})")
                        except Exception as e:
                            print(f"⚠️  Failed with {service_name}: {str(e)[:150]}")
                            import traceback
                            print(f"🔍 {service_name} error details: {traceback.format_exc()[:300]}")
                            continue
                finally:
                    browser.close()
                    
        except Exception as e:
            print(f"⚠️  Sync Playwright failed: {str(e)}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return None

    def _fetch_async_new_loop(self, video_url: str) -> Optional[str]:
        """Create a new event loop for async operations with proper error handling."""
        # Set event loop policy for Windows to avoid subprocess issues
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create the task and run it
            task = loop.create_task(self._fetch_async(video_url))
            result = loop.run_until_complete(task)
            return result
        except Exception as e:
            print(f"⚠️  Async Playwright failed: {e}")
            return None
        finally:
            try:
                # Cancel any pending tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    if not task.done():
                        task.cancel()
                        try:
                            loop.run_until_complete(task)
                        except asyncio.CancelledError:
                            pass
                        except Exception:
                            pass
            except Exception:
                pass
            finally:
                loop.close()

    async def _fetch_async(self, video_url: str) -> Optional[str]:
        """
        Async version - launches one browser instance and tries multiple scrapers.
        """
        scrapers = [
            ("youtubetotranscript.com", self._scrape_youtubetotranscript),
            ("tactiq.io", self._scrape_tactiq),
            ("kome.ai", self._scrape_komeai),
        ]

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            for service_name, scraper_func in scrapers:
                try:
                    print(f"🌐 Trying {service_name}...")
                    transcript = await scraper_func(video_url, browser)
                    if transcript and len(transcript) > 100:
                        print(f"✅ Successfully scraped from {service_name}")
                        await browser.close()
                        return transcript
                except Exception as e:
                    print(f"⚠️  Failed with {service_name}: {str(e)[:150]}")
                    continue
            await browser.close()
        return None

    # ============= SYNC VERSIONS =============

    def _handle_cookie_banner_sync(self, page):
        """Sync version: Pre-emptively clicks common cookie consent buttons."""
        consent_selectors = [
            'button:has-text("Accept")',
            'button:has-text("Accept all")',
            'button:has-text("I agree")',
            'button:has-text("Allow all")',
        ]
        try:
            for selector in consent_selectors:
                button = page.locator(selector)
                if button.is_visible(timeout=3000):
                    button.click()
                    print(" ✔ Handled cookie banner.")
                    return
        except:
            pass

    async def _scrape_youtubetotranscript_async(self, video_url: str, browser) -> Optional[str]:
        """Async version of youtubetotranscript.com scraper."""
        context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        await context.grant_permissions(['clipboard-read'])
        page = await context.new_page()
        try:
            await page.goto("https://youtubetotranscript.com/", timeout=30000)
            await self._handle_cookie_banner_async(page)

            input_selector = 'input[type="text"], input[name="youtube_url"]'
            await page.wait_for_selector(input_selector, timeout=10000)
            await page.fill(input_selector, video_url)
            
            await page.click('button[type="submit"]')
            
            copy_button_selector = '#copy-transcript'
            await page.wait_for_selector(copy_button_selector, timeout=20000)
            await page.click(copy_button_selector)
            
            # Small delay for clipboard
            await page.wait_for_timeout(500)
            
            transcript = await page.evaluate("navigator.clipboard.readText()")
            if transcript and len(transcript) > 50:
                return self._clean_text(transcript)
        finally:
            await context.close()
        return None

    async def _scrape_tactiq_async(self, video_url: str, browser) -> Optional[str]:
        """Async version of tactiq.io scraper."""
        context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        await context.grant_permissions(['clipboard-read'])
        page = await context.new_page()
        try:
            await page.goto("https://tactiq.io/tools/youtube-transcript", timeout=30000)
            await self._handle_cookie_banner_async(page)

            input_selector = 'input[placeholder*="YouTube"], input[name="url"]'
            await page.wait_for_selector(input_selector, timeout=10000)
            await page.fill(input_selector, video_url)
            
            await page.click('button:has-text("Get Transcript"), button[type="submit"]')
            
            copy_button_selector = 'button:has-text("Copy"), [data-testid="copy-button"]'
            await page.wait_for_selector(copy_button_selector, timeout=20000)
            await page.click(copy_button_selector)
            
            await page.wait_for_timeout(500)
            
            transcript = await page.evaluate("navigator.clipboard.readText()")
            if transcript and len(transcript) > 50:
                return self._clean_text(transcript)
        finally:
            await context.close()
        return None

    async def _scrape_komeai_async(self, video_url: str, browser) -> Optional[str]:
        """Async version of kome.ai scraper."""
        context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        await context.grant_permissions(['clipboard-read'])
        page = await context.new_page()
        try:
            await page.goto("https://kome.ai/tools/youtube-transcript-generator", timeout=30000)
            await self._handle_cookie_banner_async(page)

            input_selector = 'input[placeholder*="YouTube"], textarea, input[type="url"]'
            await page.wait_for_selector(input_selector, timeout=10000)
            await page.fill(input_selector, video_url)
            
            await page.click('button:has-text("Generate"), button[type="submit"]')
            
            transcript_selector = '[data-testid="transcript"], .transcript-content, pre'
            await page.wait_for_selector(transcript_selector, timeout=30000)
            
            transcript = await page.text_content(transcript_selector)
            if transcript and len(transcript) > 50:
                return self._clean_text(transcript)
        finally:
            await context.close()
        return None

    async def _handle_cookie_banner_async(self, page):
        """Async version of cookie banner handler."""
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

    def _scrape_youtubetotranscript_sync(self, video_url: str, browser) -> Optional[str]:
        """Sync version of youtubetotranscript.com scraper."""
        context = browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        context.grant_permissions(['clipboard-read'])
        page = context.new_page()
        try:
            page.goto("https://youtubetotranscript.com/", timeout=30000)
            self._handle_cookie_banner_sync(page)

            input_selector = 'input[type="text"], input[name="youtube_url"]'
            page.wait_for_selector(input_selector, timeout=10000)
            page.fill(input_selector, video_url)
            
            page.click('button[type="submit"]')
            
            copy_button_selector = '#copy-transcript'
            page.wait_for_selector(copy_button_selector, timeout=20000)
            page.click(copy_button_selector)
            
            # Small delay for clipboard
            page.wait_for_timeout(500)
            
            transcript = page.evaluate("navigator.clipboard.readText()")
            if transcript and len(transcript) > 50:
                return self._clean_text(transcript)
        finally:
            context.close()
        return None

    def _scrape_tactiq_sync(self, video_url: str, browser) -> Optional[str]:
        """Sync version of tactiq.io scraper."""
        context = browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        context.grant_permissions(['clipboard-read'])
        page = context.new_page()
        try:
            page.goto("https://tactiq.io/tools/youtube-transcript", timeout=30000)
            self._handle_cookie_banner_sync(page)

            page.fill("input[name='yt'], input[placeholder*='Enter YouTube URL']", video_url)
            page.click("input[type='submit']")
            page.wait_for_selector("div#transcript", timeout=20000)

            copy_button_selector = 'a#copy'
            page.wait_for_selector(copy_button_selector, timeout=20000)
            page.click(copy_button_selector)
            
            page.wait_for_timeout(500)
            
            transcript = page.evaluate("navigator.clipboard.readText()")
            if transcript and len(transcript) > 50:
                return self._clean_text(transcript)
        finally:
            context.close()
        return None

    def _scrape_komeai_sync(self, video_url: str, browser) -> Optional[str]:
        """Sync version of kome.ai scraper."""
        context = browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        context.grant_permissions(['clipboard-read'])
        page = context.new_page()
        try:
            page.goto("https://kome.ai/tools/youtube-transcript-generator", timeout=30000)
            self._handle_cookie_banner_sync(page)

            page.fill('input[name="url"]', video_url)
            page.click('button[type="submit"]')

            page.wait_for_selector("div.form_transcript__gnYky", timeout=20000)

            copy_button_selector = 'button.form_copyButton__KsbsH'
            page.wait_for_selector(copy_button_selector, timeout=20000)
            page.click(copy_button_selector)
            
            page.wait_for_timeout(500)
            
            transcript = page.evaluate("navigator.clipboard.readText()")
            if transcript and len(transcript) > 50:
                return self._clean_text(transcript)
        finally:
            context.close()
        return None

    # ============= ASYNC VERSIONS (unchanged) =============

    async def _handle_cookie_banner(self, page):
        """Async version: Pre-emptively clicks common cookie consent buttons."""
        consent_buttons = [
            page.locator('button:has-text("Accept")'),
            page.locator('button:has-text("Accept all")'),
            page.locator('button:has-text("I agree")'),
            page.locator('button:has-text("Allow all")'),
        ]
        try:
            for button in consent_buttons:
                if await button.is_visible(timeout=3000):
                    await button.click()
                    print(" ✔ Handled cookie banner.")
                    return
        except PlaywrightTimeoutError:
            pass

    async def _scrape_youtubetotranscript(self, video_url: str, browser: Browser) -> Optional[str]:
        """Async version of youtubetotranscript.com scraper."""
        context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        await context.grant_permissions(['clipboard-read'])
        page = await context.new_page()
        try:
            await page.goto("https://youtubetotranscript.com/", timeout=30000)
            await self._handle_cookie_banner(page)

            input_selector = 'input[type="text"], input[name="youtube_url"]'
            await page.wait_for_selector(input_selector, timeout=10000)
            await page.fill(input_selector, video_url)
            
            await page.click('button[type="submit"]')
            
            copy_button_selector = '#copy-transcript'
            await page.wait_for_selector(copy_button_selector, timeout=20000)
            await page.click(copy_button_selector)
            
            await page.wait_for_timeout(500)
            
            transcript = await page.evaluate("navigator.clipboard.readText()")
            if transcript and len(transcript) > 50:
                return self._clean_text(transcript)
        finally:
            await context.close()
        return None

    async def _scrape_tactiq(self, video_url: str, browser: Browser) -> Optional[str]:
        """Async version of tactiq.io scraper."""
        context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        await context.grant_permissions(['clipboard-read'])
        page = await context.new_page()
        try:
            await page.goto("https://tactiq.io/tools/youtube-transcript", timeout=30000)
            await self._handle_cookie_banner(page)

            await page.fill("input[name='yt'], input[placeholder*='Enter YouTube URL']", video_url)
            await page.click("input[type='submit']")
            await page.wait_for_selector("div#transcript", timeout=20000)

            copy_button_selector = 'a#copy'
            await page.wait_for_selector(copy_button_selector, timeout=20000)
            await page.click(copy_button_selector)
            
            await page.wait_for_timeout(500)
            
            transcript = await page.evaluate("navigator.clipboard.readText()")
            if transcript and len(transcript) > 50:
                return self._clean_text(transcript)
        finally:
            await context.close()
        return None

    async def _scrape_komeai(self, video_url: str, browser: Browser) -> Optional[str]:
        """Async version of kome.ai scraper."""
        context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        await context.grant_permissions(['clipboard-read'])
        page = await context.new_page()
        try:
            await page.goto("https://kome.ai/tools/youtube-transcript-generator", timeout=30000)
            await self._handle_cookie_banner(page)

            await page.fill('input[name="url"]', video_url)
            await page.click('button[type="submit"]')

            await page.wait_for_selector("div.form_transcript__gnYky", timeout=20000)

            copy_button_selector = 'button.form_copyButton__KsbsH'
            await page.wait_for_selector(copy_button_selector, timeout=20000)
            await page.click(copy_button_selector)
            
            await page.wait_for_timeout(500)
            
            transcript = await page.evaluate("navigator.clipboard.readText()")
            if transcript and len(transcript) > 50:
                return self._clean_text(transcript)
        finally:
            await context.close()
        return None

    # ============= UTILITY METHODS =============

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        text = re.sub(r'^\d{1,2}:\d{2}\s', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        return text.strip()

    def _clean_srt_format(self, text: str) -> str:
        """Clean SRT/VTT formatted text."""
        lines = []
        for line in text.splitlines():
            if '-->' in line or line.strip().isdigit() or re.match(r'^\d{2}:\d{2}', line) or 'WEBVTT' in line:
                continue
            cleaned = line.strip()
            if cleaned and not cleaned.startswith('[') and not cleaned.startswith('('):
                lines.append(cleaned)

        seen = set()
        unique_lines = [x for x in lines if not (x in seen or seen.add(x))]
        return ' '.join(unique_lines)

    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Test block
if __name__ == "__main__":
    TEST_URL = "https://www.youtube.com/watch?v=XNratwOrSiY"
    print(f"--- Running Test ---")
    print(f"Fetching transcript for: {TEST_URL}")
    
    scraper = WebTranscriptScraper()
    transcript = scraper.fetch(TEST_URL)
    
    if transcript:
        print("\n--- ✅ SUCCESS ---")
        print("Transcript (first 300 chars):")
        print(transcript[:300] + "...")
    else:
        print("\n--- ❌ FAILED ---")
        print("Could not retrieve transcript.")