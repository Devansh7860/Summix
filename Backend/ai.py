from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnablePick
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pytube import Playlist
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
load_dotenv()
import os
import asyncio
import threading

from transcript_fetcher import TranscriptFetcher

# Initialize transcript fetcher with centralized caching
fetcher = TranscriptFetcher(rate_limit_delay=3)
get_transcript = fetcher.get_transcript



# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("INDEX_NAME")
index = pc.Index(index_name)

# Legacy vector store initialization (used for Q&A functionality)
embeddings_model = HuggingFaceEndpointEmbeddings(
    repo_id="BAAI/bge-small-en-v1.5", 
    task="feature-extraction"
)
vectorstore = PineconeVectorStore(
    index = index,
    embedding = embeddings_model
)

parser = StrOutputParser() 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=150)

def create_model(api_key, model_name="gemini-2.5-flash"):
    """Create a Gemini model instance with the provided API key."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key
    )

def translate_hindi_to_english(hindi_text, api_key):
    print("Translating Hindi text to English...")
    """Translate Hindi text to English using the LLM"""
    model = create_model(api_key, "gemini-2.5-flash-lite")
    
    translation_prompt = PromptTemplate(
        template="""You are a professional translator. Translate the following Hindi text to English. 
        Maintain the original meaning and context. Only provide the English translation, no additional text.
        
        Hindi text: {hindi_text}""",
        input_variables=["hindi_text"]
    )
    
    translation_chain = translation_prompt | model | parser
    english_translation = translation_chain.invoke({"hindi_text": hindi_text})
    print("Translation complete.")
    return english_translation

def fetch_transcript(video_id, api_key, browserless_api_key=None):
    print(f"Fetching transcript for video ID: {video_id}")
    try:
        # Create a TranscriptFetcher instance with the provided Browserless API key
        if browserless_api_key:
            temp_fetcher = TranscriptFetcher(rate_limit_delay=3, browserless_api_key=browserless_api_key)
            transcript_func = temp_fetcher.get_transcript
        else:
            # Use the global fetcher as fallback
            transcript_func = get_transcript
            
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        print("Using get_transcript to fetch the transcript...")
        transcript = transcript_func(video_url)
        
        if not transcript:
            print("No transcript available for this video.")
            return None
        
        # Simple detection for Hindi script
        hindi_chars = set('अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह')
        if any(char in hindi_chars for char in transcript[:1000]):
            print("Hindi transcript detected, translating...")
            english_transcript = translate_hindi_to_english(transcript, api_key)
            print("Translation successful!")
            return english_transcript
        
        print("Transcript retrieved successfully!")
        return transcript
        
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

prompt = PromptTemplate(
    template = """ You are an AI assistant that summarizes YouTube videos based on their transcripts.  
        Treat the user like a bro. keep language semi formal but friendly and chill. 

Instructions:
- Write the summary in clear, simple language.  
- Highlight the main topics and ideas, not filler words or repetition.  
- If timestamps are available, include them with key points.  
- Organize the summary into sections (e.g., Introduction, Key Ideas, Conclusion).  
- Keep it concise, but detailed enough so the reader understands the video without watching it.  
- Do not hallucinate extra information beyond the transcript.  

Now summarize the following transcript:
        Context: {context} """,
    input_variables=["context"]
)

# Task tracking
active_tasks = {}
task_timestamps = {}  # Track when tasks were started
task_lock = threading.Lock()
TASK_TIMEOUT = 1800  # 30 minutes timeout for abandoned tasks

def cleanup_abandoned_tasks():
    """Clean up tasks that have been running for too long"""
    import time
    current_time = time.time()
    
    with task_lock:
        tasks_to_remove = []
        for task_key, start_time in task_timestamps.items():
            if current_time - start_time > TASK_TIMEOUT:
                print(f"🧹 Cleaning up abandoned task: {task_key} (running for {(current_time - start_time)/60:.1f} minutes)")
                tasks_to_remove.append(task_key)
        
        for task_key in tasks_to_remove:
            if task_key in active_tasks:
                del active_tasks[task_key]
            if task_key in task_timestamps:
                del task_timestamps[task_key]

def cancel_task(userId, contentId, is_playlist=False):
    """Cancel an active task for a user"""
    task_key = f"{userId}_{contentId}_{is_playlist}"
    
    with task_lock:
        if task_key in active_tasks:
            active_tasks[task_key] = False
            # Remove timestamp since task is being cancelled
            if task_key in task_timestamps:
                del task_timestamps[task_key]
            print(f"Cancellation requested for task: {task_key}")
            return True
        return False

def is_task_cancelled(userId, contentId, is_playlist=False):
    """Check if a task has been cancelled"""
    # First clean up any abandoned tasks
    cleanup_abandoned_tasks()
    
    task_key = f"{userId}_{contentId}_{is_playlist}"
    
    with task_lock:
        # If task isn't being tracked or is marked False, it's cancelled
        return task_key not in active_tasks or not active_tasks[task_key]

def register_task(userId, contentId, is_playlist=False):
    """Register a new active task"""
    import time
    task_key = f"{userId}_{contentId}_{is_playlist}"
    
    with task_lock:
        # Clean up old tasks before registering new one
        cleanup_abandoned_tasks()
        
        active_tasks[task_key] = True
        task_timestamps[task_key] = time.time()
        print(f"Registered new task: {task_key}")

def unregister_task(userId, contentId, is_playlist=False):
    """Unregister a completed task"""
    task_key = f"{userId}_{contentId}_{is_playlist}"
    
    with task_lock:
        if task_key in active_tasks:
            del active_tasks[task_key]
        if task_key in task_timestamps:
            del task_timestamps[task_key]
        print(f"Unregistered task: {task_key}")

def unregister_task(userId, contentId, is_playlist=False):
    """Remove a task from tracking"""
    task_key = f"{userId}_{contentId}_{is_playlist}"
    
    with task_lock:
        if task_key in active_tasks:
            del active_tasks[task_key]
            print(f"Unregistered task: {task_key}")

# main function to summarize video
def summarize(video_id, userId, api_key, browserless_api_key=None):
    print(f"Starting summarization for video: {video_id}, user: {userId}")
    
    # Register this task
    register_task(userId, video_id)
    
    try:
        # Check if summary already exists in centralized vector store
        cached_summary = fetcher._get_cached_summary_from_vector_store(video_id=video_id)
        if cached_summary:
            print("📦 Found cached summary in centralized vector store")
            unregister_task(userId, video_id)
            return cached_summary

        transcript = fetch_transcript(video_id, api_key, browserless_api_key)
        if not transcript:
            unregister_task(userId, video_id)
            print("No transcript available, cannot generate summary")
            return "No transcript available for this video."

        # Check for cancellation after expensive operation
        if is_task_cancelled(userId, video_id):
            print(f"Task cancelled: summarize video {video_id} for user {userId}")
            return "Task was cancelled"

        print("Generating summary...")
        model = create_model(api_key)
        chain = prompt | model | parser
        summary = chain.invoke({"context": transcript})
        print("Summary generated successfully!")
        
        # Check for cancellation again
        if is_task_cancelled(userId, video_id):
            print(f"Task cancelled after summary generation: {video_id}")
            return "Task was cancelled"
            
        # Cache summary to centralized vector store
        print("💾 Caching summary to centralized vector store...")
        fetcher._cache_summary_to_vector_store(summary, video_id=video_id)
        
        # Unregister completed task
        unregister_task(userId, video_id)
        return summary
    except Exception as e:
        # Make sure to unregister on error
        unregister_task(userId, video_id)
        print(f"Error in summarize: {e}")
        raise e



def ask(video_id, playlist_id, question, userId, api_key):
    """
    Answer questions about videos or playlists using centralized vector storage.
    
    For videos: Retrieves transcript from centralized cache and uses it as context
    For playlists: Retrieves individual video summaries and combines them as context
    """
    print(f"Asking question: {question}")
    print(f"Video ID: {video_id}, Playlist ID: {playlist_id}")
    
    try:
        context = ""
        
        if video_id and not playlist_id:
            # Handle individual video questions
            print(f"Retrieving transcript for video: {video_id}")
            
            # Get transcript from centralized vector store
            transcript = fetcher._get_cached_transcript_from_vector_store(video_id)
            
            if not transcript:
                # If not cached, try to fetch it
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                transcript = get_transcript(video_url)
                
            if transcript:
                context = transcript
                print(f"Using transcript context (length: {len(context)} chars)")
            else:
                return "Sorry, I couldn't find the transcript for this video. Please make sure the video has been processed first."
                
        elif playlist_id and not video_id:
            # Handle playlist questions
            print(f"Retrieving summaries for playlist: {playlist_id}")
            
            # Get playlist summary from centralized vector store
            playlist_summary = fetcher._get_cached_summary_from_vector_store(playlist_id=playlist_id, video_id=False)
            
            if playlist_summary:
                context = playlist_summary
                print(f"Using playlist summary context (length: {len(context)} chars)")
            else:
                return "Sorry, I couldn't find the summary for this playlist. Please make sure the playlist has been processed first."
                
        elif video_id and playlist_id:
            # Handle specific video within playlist context
            print(f"Retrieving summary for video {video_id} in playlist {playlist_id}")
            
            # First try to get individual video summary
            video_summary = fetcher._get_cached_summary_from_vector_store(video_id=video_id)
            
            if video_summary:
                context = video_summary
                print(f"Using video summary context (length: {len(context)} chars)")
            else:
                # Fallback to transcript if summary not available
                transcript = fetcher._get_cached_transcript_from_vector_store(video_id)
                if transcript:
                    context = transcript
                    print(f"Using transcript context (length: {len(context)} chars)")
                else:
                    return "Sorry, I couldn't find information for this video. Please make sure it has been processed first."
        else:
            return "Please provide either a video ID or playlist ID to ask questions about."
        
        if not context:
            return "Sorry, I couldn't find relevant content to answer your question."
        
        # Create the prompt for answering questions
        prompt = PromptTemplate(
            template="""You are an AI assistant that answers questions about YouTube videos based on their transcripts or summaries.
            If the answer is not present in the context, say "I don't know based on the provided content."
            Keep your language friendly and conversational.
            
            Context: {context}
            
            Question: {question}
            
            Answer:""",
            input_variables=["context", "question"]
        )
        
        # Create and run the chain
        model = create_model(api_key)
        chain = prompt | model | parser
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        return answer
        
    except Exception as e:
        print(f"Error in ask function: {e}")
        return "Sorry, I encountered an error while processing your question. Please try again."



# ============================================================================
# STREAMING PIPELINE INFRASTRUCTURE
# ============================================================================

class VideoData:
    """Container for video data flowing through the pipeline"""
    def __init__(self, video_id, playlist_id, userId, api_key):
        self.video_id = video_id
        self.playlist_id = playlist_id
        self.userId = userId  
        self.api_key = api_key
        self.transcript = None
        self.summary = None
        self.error = None
        self.cached = False

async def translate_stage_worker(input_queue, output_queue, userId, playlist_id, api_key):
    """Worker for translating transcripts (if needed)"""
    while True:
        video_data = await input_queue.get()
        if video_data is None:  # Sentinel value to stop worker
            input_queue.task_done()
            break
            
        try:
            # Check cancellation before processing
            if is_task_cancelled(userId, playlist_id, is_playlist=True):
                video_data.error = "Task cancelled during translation"
                await output_queue.put(video_data)
                input_queue.task_done()
                continue
                
            # Skip if there's an error from previous stage
            if video_data.error:
                await output_queue.put(video_data)
                input_queue.task_done()
                continue
                
            print(f"🔄 Processing transcript for video {video_data.video_id} (translation stage)")
            
            # Check if Hindi translation is needed
            if video_data.transcript:
                # Simple detection for Hindi script
                hindi_chars = set('अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह')
                if any(char in hindi_chars for char in video_data.transcript[:1000]):
                    print(f"Hindi transcript detected for video ID: {video_data.video_id}, translating...")
                    try:
                        def _translate_sync():
                            return translate_hindi_to_english(video_data.transcript, api_key)
                        
                        translated_transcript = await asyncio.to_thread(_translate_sync)
                        video_data.transcript = translated_transcript
                        print(f"✅ Successfully translated transcript for video {video_data.video_id}")
                    except Exception as e:
                        print(f"❌ Error translating transcript for {video_data.video_id}: {e}")
                        # Continue with original transcript if translation fails
                        
        except Exception as e:
            video_data.error = f"Error in translation stage: {e}"
            print(f"❌ Error in translation stage for {video_data.video_id}: {e}")
            
        await output_queue.put(video_data)
        input_queue.task_done()

async def summarize_stage_worker(input_queue, output_queue, userId, playlist_id, api_key):
    """Worker for generating summaries"""
    while True:
        video_data = await input_queue.get()
        if video_data is None:  # Sentinel value to stop worker
            input_queue.task_done()
            break
            
        try:
            # Check cancellation before processing
            if is_task_cancelled(userId, playlist_id, is_playlist=True):
                video_data.error = "Task cancelled during summarization"
                await output_queue.put(video_data)
                input_queue.task_done()
                continue
                
            # Skip if there's an error from previous stage
            if video_data.error:
                await output_queue.put(video_data)
                input_queue.task_done()
                continue
                
            print(f"🔄 Generating summary for video {video_data.video_id}")
            
            # Check cache first
            cached_summary = fetcher._get_cached_summary_from_vector_store(video_id=video_data.video_id)
            if cached_summary:
                print(f"📦 Using cached summary for video {video_data.video_id}")
                video_data.summary = cached_summary
                video_data.cached = True
            else:
                # Generate new summary
                prompt = PromptTemplate(
                    template="""You are an expert summarizer and teacher. treat user like a bro. keep language semi formal but friendly and chill.

I'll provide you with a transcript from a YouTube video. Your task is to create a comprehensive summary that includes:

1. **Main Topic & Purpose**: What is this video about and what is its primary goal?

2. **Key Concepts & Ideas**: List the main concepts, ideas, or topics covered (use bullet points)

3. **Important Details**: Any specific facts, examples, or explanations that are crucial for understanding

4. **Practical Takeaways**: What should someone remember or apply after watching this video?

Please make sure your summary is:
- Clear and well-structured  
- Comprehensive but concise
- Educational and easy to understand
- Written in a friendly, approachable tone

Transcript: {context}""",
                    input_variables=["context"]
                )
                
                model = create_model(api_key)
                chain = prompt | model | parser
                summary = await asyncio.to_thread(chain.invoke, {"context": video_data.transcript})
                
                # Cache summary to centralized vector store
                fetcher._cache_summary_to_vector_store(summary, video_id=video_data.video_id, playlist_id=playlist_id)
                video_data.summary = summary
                print(f"✅ Successfully generated summary for video {video_data.video_id}")
                
        except Exception as e:
            video_data.error = f"Error generating summary: {e}"
            print(f"❌ Error generating summary for {video_data.video_id}: {e}")
            
        await output_queue.put(video_data)
        input_queue.task_done()

async def streaming_playlist_pipeline(playlist_id, userId, api_key, max_concurrent=3, browserless_api_key=None):
    """
    Hybrid streaming pipeline that uses optimized batch fetching + streaming processing
    Stage 1: Optimized concurrent transcript fetching (batch)  
    Stage 2: Streaming translate → summarize → collect
    """
    print(f"🚀 Starting hybrid streaming pipeline for playlist {playlist_id}")
    
    # Check cancellation at the very start
    if is_task_cancelled(userId, playlist_id, is_playlist=True):
        print(f"Task cancelled before starting playlist {playlist_id}")
        return "Task was cancelled"
    
    # STAGE 1: Use optimized batch transcript fetching (already concurrent & optimized)
    try:
        playlist = Playlist(f"https://www.youtube.com/playlist?list={playlist_id}")
        video_urls = list(playlist.video_urls)
        total_videos = len(video_urls)
        
        if total_videos == 0:
            print("❌ No videos found in playlist")
            return None
            
        print(f"📋 Found {total_videos} videos in playlist")
        
        # Check cancellation after getting video list
        if is_task_cancelled(userId, playlist_id, is_playlist=True):
            print(f"Task cancelled after getting video list for playlist {playlist_id}")
            return "Task was cancelled"
            
        print(f"🔄 Stage 1: Fetching all transcripts using optimized batch method...")
        
        # Use the existing optimized batch fetching
        def _fetch_transcripts_batch():
            # Create a TranscriptFetcher instance with the provided Browserless API key
            if browserless_api_key:
                temp_fetcher = TranscriptFetcher(rate_limit_delay=3, browserless_api_key=browserless_api_key)
                return temp_fetcher.get_playlist_transcripts(video_urls)
            else:
                # Use the global fetcher as fallback
                return fetcher.get_playlist_transcripts(video_urls)
        
        transcript_results = await asyncio.to_thread(_fetch_transcripts_batch)
        
        # Check cancellation after transcript fetching
        if is_task_cancelled(userId, playlist_id, is_playlist=True):
            print(f"Task cancelled after transcript fetching for playlist {playlist_id}")
            return "Task was cancelled"
        
        # Convert results to VideoData objects with transcripts
        video_data_list = []
        for video_url, transcript in transcript_results.items():
            # Check cancellation during video processing
            if is_task_cancelled(userId, playlist_id, is_playlist=True):
                print(f"Task cancelled during video data processing for playlist {playlist_id}")
                return "Task was cancelled"
                
            video_id = video_url.split("v=")[1].split("&")[0] if "v=" in video_url else video_url.split("/")[-1]
            video_data = VideoData(video_id, playlist_id, userId, api_key)
            
            if transcript:
                video_data.transcript = transcript
                video_data_list.append(video_data)
                print(f"✅ Transcript ready for video {video_id}")
            else:
                video_data.error = f"No transcript available for video {video_id}"
                video_data_list.append(video_data)
                print(f"❌ No transcript for video {video_id}")
        
        print(f"📊 Stage 1 complete: {len([v for v in video_data_list if v.transcript])}/{total_videos} transcripts fetched")
        
    except Exception as e:
        print(f"❌ Error in batch transcript fetching: {e}")
        return None
    
    # Check cancellation before starting Stage 2
    if is_task_cancelled(userId, playlist_id, is_playlist=True):
        print(f"Task cancelled before Stage 2 for playlist {playlist_id}")
        return "Task was cancelled"
    
    # STAGE 2: Streaming processing pipeline (translate → summarize → collect)
    print(f"🔄 Stage 2: Starting streaming processing pipeline...")
    
    translate_queue = asyncio.Queue()
    summarize_queue = asyncio.Queue() 
    results_queue = asyncio.Queue()
    
    # Add all video data to translate queue
    for video_data in video_data_list:
        await translate_queue.put(video_data)
    
    # Start worker tasks for streaming stages
    num_workers = min(max_concurrent, len(video_data_list))
    
    # Translate stage workers  
    translate_workers = []
    for i in range(num_workers):
        worker = asyncio.create_task(
            translate_stage_worker(translate_queue, summarize_queue, userId, playlist_id, api_key)
        )
        translate_workers.append(worker)
    
    # Summarize stage workers
    summarize_workers = []
    for i in range(num_workers):
        worker = asyncio.create_task(
            summarize_stage_worker(summarize_queue, results_queue, userId, playlist_id, api_key)
        )
        summarize_workers.append(worker)
    
    # Collect results from streaming processing
    completed_videos = []
    videos_processed = 0
    
    try:
        while videos_processed < len(video_data_list):
            # Check for cancellation
            if is_task_cancelled(userId, playlist_id, is_playlist=True):
                print("❌ Task cancelled during streaming processing")
                break
                
            # Get completed video
            video_data = await results_queue.get()
            videos_processed += 1
            
            if video_data.error:
                print(f"❌ Video {video_data.video_id} failed: {video_data.error}")
            elif video_data.summary:
                completed_videos.append(video_data)
                print(f"✅ Completed video {video_data.video_id} ({videos_processed}/{len(video_data_list)})")
            
            results_queue.task_done()
            
        print(f"📊 Stage 2 complete: {len(completed_videos)}/{len(video_data_list)} videos processed successfully")
        
    finally:
        # Stop all workers by sending sentinel values
        for _ in range(num_workers):
            await translate_queue.put(None)
            await summarize_queue.put(None)
            
        # Wait for all workers to finish
        await asyncio.gather(*translate_workers, *summarize_workers, return_exceptions=True)
        
    return completed_videos

async def generate_final_playlist_summary(completed_videos, playlist_id, api_key):
    """Generate final playlist summary from completed video summaries"""
    if not completed_videos:
        return "No videos were successfully processed."
        
    print(f"📋 Generating final playlist summary from {len(completed_videos)} videos")
    
    # Combine all individual summaries
    combined_summaries = "\n\n".join([
        f"Video {i+1}: {video_data.summary}" 
        for i, video_data in enumerate(completed_videos)
    ])
    
    # Create final summary prompt
    prompt2 = PromptTemplate(
        template="""You are an expert summarizer and teacher. 
I will provide you with summaries of multiple videos from a playlist.treat user like a bro. keep language semi formal but friendly and chill.

Your task is to generate a structured final summary in two parts:

1. **Overall Playlist Summary**  
   - Provide a cohesive explanation of the entire playlist as a whole.  
   - Highlight the main recurring themes, the flow of ideas across videos, 
     and the overall knowledge or story the playlist conveys.Do not hallucinate extra information beyond the provided summaries.
   - Keep this section clear and concise, as if you are explaining to someone 
     who has not seen the playlist but wants to grasp the big picture.  

2. **Video-wise Key Concepts**  
   - For each video, list its key takeaways in bullet points.  
   - Use the format:  
     - *Video 1 (Title/Number):* [3-5 key points]  
     - *Video 2 (Title/Number):* [3-5 key points]  
   - Make sure to keep the language simple, precise, and educational.  

summaries: {combined_summaries}""",
        input_variables=["combined_summaries"]
    )

    # Generate final summary
    final_model = create_model(api_key)
    final_chain = prompt2 | final_model | parser
    final_summary = await asyncio.to_thread(final_chain.invoke, {"combined_summaries": combined_summaries})
    
    # Cache playlist summary to centralized vector store
    print("💾 Caching playlist summary to centralized vector store...")
    fetcher._cache_summary_to_vector_store(final_summary, playlist_id=playlist_id)
    
    return final_summary

async def summarizePlaylist(playlist_id, userId, api_key, browserless_api_key=None):
    """
    Streaming playlist summarizer that processes videos through concurrent pipeline stages
    Each video flows through: fetch → translate → summarize → collect
    """
    print("🚀 Starting streaming playlist summarization:", playlist_id)
    
    # Register this task
    register_task(userId, playlist_id, is_playlist=True)
    
    try:
        # Check if playlist summary already exists in centralized vector store
        cached_summary = fetcher._get_cached_summary_from_vector_store(playlist_id=playlist_id)
        if cached_summary:
            print("📦 Found cached playlist summary in centralized vector store")
            unregister_task(userId, playlist_id, is_playlist=True)
            return cached_summary
        
        # Check for cancellation before starting pipeline
        if is_task_cancelled(userId, playlist_id, is_playlist=True):
            print(f"Task cancelled: summarize playlist {playlist_id} - before starting pipeline")
            unregister_task(userId, playlist_id, is_playlist=True)
            return "Task was cancelled"
        
        # Run streaming pipeline
        print("🔄 Starting streaming pipeline...")
        completed_videos = await streaming_playlist_pipeline(playlist_id, userId, api_key, max_concurrent=3, browserless_api_key=browserless_api_key)
        
        # Check for cancellation after pipeline
        if is_task_cancelled(userId, playlist_id, is_playlist=True):
            print(f"Task cancelled: summarize playlist {playlist_id} - after pipeline")
            unregister_task(userId, playlist_id, is_playlist=True)
            return "Task was cancelled"
        
        # Handle case where no videos were processed successfully
        if not completed_videos:
            unregister_task(userId, playlist_id, is_playlist=True)
            return "No videos could be processed successfully from this playlist."
        
        # Generate final playlist summary
        print("🔄 Generating final playlist summary...")
        final_summary = await generate_final_playlist_summary(completed_videos, playlist_id, api_key)
        
        # Final cancellation check
        if is_task_cancelled(userId, playlist_id, is_playlist=True):
            print(f"Task cancelled: summarize playlist {playlist_id} - after final summary")
            unregister_task(userId, playlist_id, is_playlist=True)
            return "Task was cancelled"
        
        # Unregister completed task
        unregister_task(userId, playlist_id, is_playlist=True)
        print(f"✅ Playlist summarization completed successfully!")
        return final_summary
        
    except Exception as e:
        # Make sure to unregister on error
        unregister_task(userId, playlist_id, is_playlist=True)
        print(f"❌ Error in summarizePlaylist: {e}")
        raise e
 
