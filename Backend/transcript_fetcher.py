
"""
Main transcript fetcher module that coordinates different fetching methods.
Uses centralized vector storage instead of local file caching.
"""
import os
import time
from typing import Optional, List, Dict
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

from api_fetcher import APITranscriptFetcher
# from ytdlp_fetcher import YtDlpTranscriptFetcher
from web_scraper import WebTranscriptScraper
from utils import extract_video_id, clean_transcript_text


class TranscriptFetcher:
    def __init__(self, rate_limit_delay=2):
        """
        Initialize the transcript fetcher with multiple backend methods.
        Uses centralized vector storage for caching.
        
        Args:
            rate_limit_delay: Minimum delay between requests (in seconds)
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
        # Initialize different fetchers
        self.api_fetcher = APITranscriptFetcher()
        # self.ytdlp_fetcher = YtDlpTranscriptFetcher()
        self.web_scraper = WebTranscriptScraper()
        
        # Initialize vector store for centralized caching
        self._init_vector_store()
    
    def _init_vector_store(self):
        """Initialize the vector store for centralized transcript/summary storage."""
        try:
            # Initialize Pinecone
            pc = Pinecone(
                api_key=os.getenv("PINECONE_API_KEY")
            )
            
            # Initialize embeddings
            embeddings_model = HuggingFaceEndpointEmbeddings(
                repo_id="BAAI/bge-small-en-v1.5", 
                task="feature-extraction",
            )
            
            # Initialize vector store
            self.vector_store = PineconeVectorStore(
                index=pc.Index(os.getenv("INDEX_NAME")),
                embedding=embeddings_model
            )
            print("Vector store initialized for centralized caching")
        except Exception as e:
            print(f"Could not initialize vector store: {e}")
            self.vector_store = None
    
    def get_transcript(self, video_url: str, use_cache: bool = True, method: str = "auto") -> Optional[str]:
        """
        Fetch transcript with intelligent fallback methods.
        
        Args:
            video_url: YouTube video URL
            use_cache: Whether to use cached transcripts
            method: "api", "yt_dlp", "web", or "auto" for automatic selection
        
        Returns:
            Transcript text or None if unavailable
        """
        video_id = extract_video_id(video_url)
        if not video_id:
            print("❌ Invalid YouTube URL")
            return None
        
        # Check vector store first (centralized cache)
        if use_cache:
            cached_transcript = self._get_cached_transcript_from_vector_store(video_id)
            if cached_transcript:
                print(f"📦 Using cached transcript for {video_id}")
                return cached_transcript
        
        # Enforce rate limiting
        self._enforce_rate_limit()
        
        transcript = None
        
        # Method 1: YouTube Transcript API (fastest and most reliable)
        # if method in ["auto", "api"] and self.api_fetcher.is_available():
        #     print("Trying YouTube Transcript API...")
        #     transcript = self.api_fetcher.fetch_transcript(video_id)
        #     if transcript:
        #         print("Successfully fetched via YouTube Transcript API")
        #         if use_cache:
        #             self._cache_transcript_to_vector_store(video_id, transcript, method="api")
        #         return transcript
        
        # Method 2: Web scraping (reliable fallback, skip yt-dlp to avoid 429 errors)
        if method in ["auto", "web"] and not transcript:
            print("API failed. Trying web scraping...")
            transcript = self.web_scraper.fetch(video_url)
            if transcript:
                print("Successfully fetched via web scraping")
                if use_cache:
                    self._cache_transcript_to_vector_store(video_id, transcript, method="web_scraper")
                return transcript
        
        print("Failed to fetch transcript using all available methods")
        return None
    
    def get_playlist_transcripts(self, video_urls: List[str], use_cache: bool = True) -> Dict[str, Optional[str]]:
        """
        Efficiently fetch transcripts for multiple videos.
        Uses optimized web scraping with a single browser instance when needed.
        
        Args:
            video_urls: List of YouTube video URLs
            use_cache: Whether to use cached transcripts
            
        Returns:
            Dictionary mapping video URLs to their transcripts
        """
        print(f"Fetching transcripts for {len(video_urls)} videos...")
        
        results = {}
        uncached_urls = []
        
        # First, check vector store for all videos (centralized cache)
        if use_cache:
            for url in video_urls:
                video_id = extract_video_id(url)
                if video_id:
                    cached_transcript = self._get_cached_transcript_from_vector_store(video_id)
                    if cached_transcript:
                        print(f"Using cached transcript for {video_id}")
                        results[url] = cached_transcript
                    else:
                        uncached_urls.append(url)
                else:
                    results[url] = None
        else:
            uncached_urls = video_urls
        
        # If we have uncached videos, try to fetch them efficiently
        if uncached_urls:
            print(f"Fetching {len(uncached_urls)} uncached videos...")
            
            # Try API method first for all uncached videos (fastest & most reliable)
            api_results = {}
            # for url in uncached_urls:
            #     video_id = extract_video_id(url)
            #     if video_id:
            #         print(f"Trying API for {video_id}...")
            #         transcript = self.api_fetcher.fetch_transcript(video_id)
            #         if transcript:
            #             api_results[url] = transcript
            #             if use_cache:
            #                 self._cache_transcript_to_vector_store(video_id, transcript, method="api")
            #             print(f"API success for {video_id}")
            
            # Update results and find videos that still need transcripts
            results.update(api_results)
            remaining_urls = [url for url in uncached_urls if url not in api_results]
            
            # For remaining videos, use optimized web scraping (skip yt-dlp to avoid 429 errors)
            if remaining_urls:
                print(f"API failed for {len(remaining_urls)} videos. Using optimized web scraping...")
                web_results = self.web_scraper.fetch_playlist(remaining_urls)
                
                # Cache web scraping results in vector store
                if use_cache:
                    for url, transcript in web_results.items():
                        if transcript:
                            video_id = extract_video_id(url)
                            if video_id:
                                self._cache_transcript_to_vector_store(video_id, transcript, method="web_scraper")
                
                results.update(web_results)
        
        success_count = sum(1 for t in results.values() if t)
        print(f"Playlist complete: {success_count}/{len(video_urls)} successful")
        
        return results
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            print(f"⏳ Rate limiting: waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cached_transcript_from_vector_store(self, video_id: str, playlist_id: bool = False) -> Optional[str]:
        """Get cached transcript from vector store."""
        if not self.vector_store:
            return None
            
        try:
            # Search for existing transcript chunks with this video_id
            filter_dict = {
                "videoId": video_id,
                "isSummary": False
            }
            if playlist_id:
                filter_dict["playlistId"] = playlist_id
            
            # Search for transcript chunks
            results = self.vector_store.similarity_search(
                query=f"transcript for video {video_id}",
                k=20,  # Get many chunks to reconstruct full transcript
                filter=filter_dict
            )
            
            if results:
                # Reconstruct transcript from chunks
                transcript_chunks = [(doc.metadata.get('chunk_index', 0), doc.page_content) for doc in results]
                transcript_chunks.sort(key=lambda x: x[0])  # Sort by chunk index
                
                full_transcript = ' '.join([chunk[1] for chunk in transcript_chunks])
                print(f"📦 Found cached transcript for {video_id} ({len(results)} chunks)")
                return full_transcript
            
        except Exception as e:
            print(f"⚠️ Error retrieving cached transcript: {e}")
        
        return None
    
    def _cache_transcript_to_vector_store(self, video_id: str, transcript: str, method: str = "unknown", playlist_id:bool = False):
        """Cache transcript to vector store as chunks."""
        if not self.vector_store:
            return
            
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # Split transcript into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            
            chunks = text_splitter.create_documents([transcript])
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata = {
                    "videoId": video_id,
                    "playlistId": playlist_id,
                    "isSummary": False,
                    "method": method,
                    "timestamp": time.time(),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            
            # Store chunks in vector store
            self.vector_store.add_documents(chunks)
            print(f"💾 Cached transcript for {video_id} as {len(chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Error caching transcript: {e}")
    
    def _get_cached_summary_from_vector_store(self, video_id: str | bool = None, playlist_id: bool = False) -> Optional[str]:
        """Get cached summary from vector store."""
        if not self.vector_store:
            return None
            
        try:
            # Build filter for summary
            filter_dict = {"isSummary": True}
            
            if video_id:
                filter_dict["videoId"] = video_id
            if playlist_id:
                filter_dict["playlistId"] = playlist_id
            
            # Search for summary
            results = self.vector_store.similarity_search(
                query=f"summary for {'video ' + video_id if video_id else 'playlist ' + playlist_id}",
                k=1,
                filter=filter_dict
            )
            
            if results:
                print(f"📦 Found cached summary for {'video ' + video_id if video_id else 'playlist ' + playlist_id}")
                return results[0].page_content
                
        except Exception as e:
            print(f"⚠️ Error retrieving cached summary: {e}")
        
        return None
    
    def _cache_summary_to_vector_store(self, summary: str, video_id: str = None, playlist_id: bool = False):
        """Cache summary to vector store."""
        if not self.vector_store:
            return
            
        try:
            from langchain.schema import Document
            
            # Create summary document
            metadata = {
                "isSummary": True,
                "timestamp": time.time()
            }
            
            if video_id:
                metadata["videoId"] = video_id
            if playlist_id:
                metadata["playlistId"] = playlist_id
                
            doc = Document(
                page_content=summary,
                metadata=metadata
            )
            
            # Store summary in vector store
            self.vector_store.add_documents([doc])
            print(f"💾 Cached summary for {'video ' + video_id if video_id else 'playlist ' + playlist_id}")
            
        except Exception as e:
            print(f"❌ Error caching summary: {e}")


# Convenience function for backward compatibility
def get_transcript(video_url: str, use_cache: bool = True) -> Optional[str]:
    """
    Simple interface for fetching transcripts.
    
    Args:
        video_url: YouTube video URL
        use_cache: Whether to use cached transcripts
    
    Returns:
        Transcript text or None
    """
    fetcher = TranscriptFetcher(rate_limit_delay=3)
    return fetcher.get_transcript(video_url, use_cache=use_cache)


# Example usage
if __name__ == "__main__":
    print("YouTube Transcript Fetcher")
    print("=" * 50)
    
    test_url = "https://www.youtube.com/watch?v=GOejI6c0CMQ"
    
    print(f"\n📹 Fetching transcript for: {test_url}")
    transcript = get_transcript(test_url)
    
    if transcript:
        # preview = transcript[:500] if len(transcript) > 500 else transcript
        print(f"\n✅ Success! Preview:\n{transcript}...")
        # print(f"\n📊 Total length: {len(transcript)} characters")
    else:
        print("\n❌ Failed to fetch transcript")