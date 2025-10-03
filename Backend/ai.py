from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pytube import Playlist
from dotenv import load_dotenv
from utils import extract_video_id
load_dotenv()
import os
import asyncio
import threading

from transcript_fetcher import TranscriptFetcher
from websocket_task_manager import task_manager, TaskStatus

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

def fetch_transcript(video_id, api_key, browserless_api_key=None, userId=None, task_id=None):
    print(f"Fetching transcript for video ID: {video_id}")
    
    # Check cancellation before starting
    if userId and task_id:
        try:
            task_status = task_manager.get_task_status_sync(task_id)
            if task_status == TaskStatus.CANCELLED:
                print(f"Task cancelled before transcript fetch: {video_id}")
                return None
        except:
            # If task doesn't exist, continue processing
            pass
        
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
        
        # Check cancellation after transcript fetch
        if userId and task_id:
            try:
                task_status = task_manager.get_task_status_sync(task_id)
                if task_status == TaskStatus.CANCELLED:
                    print(f"Task cancelled after transcript fetch: {video_id}")
                    return None
            except:
                pass
        
        if not transcript:
            print("No transcript available for this video.")
            return None
        
        # Simple detection for Hindi script
        hindi_chars = set('अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह')
        if any(char in hindi_chars for char in transcript[:1000]):
            print("Hindi transcript detected, translating...")
            
            # Check cancellation before translation
            if userId and task_id:
                try:
                    task_status = task_manager.get_task_status_sync(task_id)
                    if task_status == TaskStatus.CANCELLED:
                        print(f"Task cancelled before Hindi translation: {video_id}")
                        return None
                except:
                    pass
                
            english_transcript = translate_hindi_to_english(transcript, api_key)
            
            # Check cancellation after translation
            if userId and task_id:
                try:
                    task_status = task_manager.get_task_status_sync(task_id)
                    if task_status == TaskStatus.CANCELLED:
                        print(f"Task cancelled after Hindi translation: {video_id}")
                        return None
                except:
                    pass
                
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

# main function to summarize video with WebSocket task management
async def summarize(video_id, userId, api_key, browserless_api_key=None, task_id=None):
    print(f"Starting summarization for video: {video_id}, user: {userId}, task: {task_id}")
    
    # Update task status to running
    if task_id:
        await task_manager.update_task(task_id, status=TaskStatus.RUNNING, progress=5, message="Starting video summarization")
    
    try:
        # Check if summary already exists in centralized vector store
        cached_summary = fetcher._get_cached_summary_from_vector_store(video_id=video_id)
        if cached_summary:
            print("📦 Found cached summary in centralized vector store")
            # Return cached summary (background task will handle WebSocket update)
            return cached_summary

        # Update progress
        if task_id:
            await task_manager.update_task(task_id, progress=10, message="Fetching video transcript")
        
        # Call fetch_transcript in a thread since it's not async
        import asyncio
        transcript = await asyncio.to_thread(fetch_transcript, video_id, api_key, browserless_api_key, userId, task_id)
        if not transcript:
            # Check if task was cancelled first - if so, don't send error message
            if task_id:
                task_status = await task_manager.get_task_status(task_id)
                if task_status == TaskStatus.CANCELLED:
                    print(f"Task cancelled during transcript fetch: {video_id}")
                    return "Task was cancelled"
            
            # Task wasn't cancelled, so it's a real transcript error
            if task_id:
                await task_manager.update_task(task_id, status=TaskStatus.FAILED, progress=100, 
                                             message="No transcript available", error="No transcript available for this video.")
            print("No transcript available, cannot generate summary")
            return "No transcript available for this video."

        # Check for cancellation after transcript
        if task_id:
            task_status = await task_manager.get_task_status(task_id)
            if task_status == TaskStatus.CANCELLED:
                print(f"Task cancelled after transcript fetch: {video_id}")
                return "Task was cancelled"

        # Update progress
        if task_id:
            await task_manager.update_task(task_id, progress=30, message="Generating AI summary")

        print("Generating summary...")
        model = create_model(api_key)
        
        # Small delay to allow cancellation requests to arrive
        import time
        time.sleep(0.1)  # 100ms delay
        
        # Check cancellation before AI processing
        if task_id:
            task_status = await task_manager.get_task_status(task_id)
            if task_status == TaskStatus.CANCELLED:
                print(f"Task cancelled before AI processing: {video_id}")
                return "Task was cancelled"
        
        chain = prompt | model | parser
        
        # For long AI processing, we'll use threading with periodic checks
        import threading
        import time
        
        summary_result = {"value": None, "error": None, "cancelled": False}
        
        def ai_worker():
            try:
                # Check for cancellation RIGHT before starting AI processing
                if task_id:
                    task_status = task_manager.get_task_status_sync(task_id)
                    if task_status == TaskStatus.CANCELLED:
                        print(f"Task cancelled before starting AI chain: {video_id}")
                        summary_result["cancelled"] = True
                        return
                    
                summary_result["value"] = chain.invoke({"context": transcript})
            except Exception as e:
                summary_result["error"] = str(e)
        
        # Start AI processing in background thread
        ai_thread = threading.Thread(target=ai_worker)
        ai_thread.start()
        
        # Wait with periodic cancellation checks (every 50ms for good balance)
        while ai_thread.is_alive():
            ai_thread.join(timeout=0.05)  # Wait max 50ms for good balance
            
            # Check for cancellation every 50ms
            if task_id:
                task_status = task_manager.get_task_status_sync(task_id)
                if task_status == TaskStatus.CANCELLED:
                    print(f"Task cancelled during AI processing: {video_id}")
                    summary_result["cancelled"] = True
                    # Thread will finish naturally, we just won't use the result
                    break
        
        # Wait for thread to finish if not cancelled
        if not summary_result["cancelled"]:
            ai_thread.join()

        # Handle results
        if summary_result["cancelled"] or (task_id and task_manager.get_task_status_sync(task_id) == TaskStatus.CANCELLED):
            print(f"Task was cancelled during or after AI processing: {video_id}")
            # Don't send WebSocket messages for cancelled tasks to avoid showing messages for old content
            # if task_id:
            #     await task_manager.update_task(task_id, status=TaskStatus.CANCELLED, progress=100, message="Task was cancelled")
            return "Task was cancelled"
            
        if summary_result["error"]:
            if task_id:
                await task_manager.update_task(task_id, status=TaskStatus.FAILED, progress=100, 
                                             message="AI processing failed", error=str(summary_result["error"]))
            raise Exception(summary_result["error"])
            
        summary = summary_result["value"]
        print("Summary generated successfully!")
        
        # Check for cancellation again before caching
        if task_id:
            task_status = await task_manager.get_task_status(task_id)
            if task_status == TaskStatus.CANCELLED:
                print(f"Task cancelled after summary generation - not caching: {video_id}")
                return "Task was cancelled"
        
        # Update progress
        if task_id:
            await task_manager.update_task(task_id, progress=90, message="Caching summary")
            
        # Cache summary to centralized vector store (only for non-cancelled tasks)
        print("💾 Caching summary to centralized vector store...")
        fetcher._cache_summary_to_vector_store(summary, video_id=video_id)
        
        # Return summary (background task will handle WebSocket update)
        return summary
    except Exception as e:
        # Make sure to update task status on error
        if task_id:
            await task_manager.update_task(task_id, status=TaskStatus.FAILED, progress=100, 
                                         message="Summarization failed", error=str(e))
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

async def streaming_playlist_pipeline(playlist_id, userId, api_key, max_concurrent=3, browserless_api_key=None, task_id=None):
    """
    TRUE STREAMING PIPELINE: Fetch and process videos individually with immediate cancellation
    No batch fetching - each video is fetched and processed as soon as possible
    """
    print(f"🚀 Starting TRUE streaming pipeline for playlist {playlist_id}")
    
    # Check cancellation at the very start
    if task_id:
        task_status = await task_manager.get_task_status(task_id)
        if task_status == TaskStatus.CANCELLED:
            print(f"Task cancelled before starting playlist {playlist_id}")
            return "Task was cancelled"
    
    # Get playlist video URLs
    try:
        playlist = Playlist(f"https://www.youtube.com/playlist?list={playlist_id}")
        video_urls = list(playlist.video_urls)
        total_videos = len(video_urls)
        
        if total_videos == 0:
            print("❌ No videos found in playlist")
            return None
            
        print(f"📋 Found {total_videos} videos in playlist")
        
        # Check cancellation after getting video list
        if task_id:
            task_status = await task_manager.get_task_status(task_id)
            if task_status == TaskStatus.CANCELLED:
                print(f"Task cancelled after getting video list for playlist {playlist_id}")
                return "Task was cancelled"
    
    except Exception as e:
        print(f"❌ Error getting playlist videos: {e}")
        return None
    
    # Initialize queues and results
    completed_videos = []
    videos_processed = 0
    
    # Create fetcher instance
    temp_fetcher = TranscriptFetcher(rate_limit_delay=2, browserless_api_key=browserless_api_key) if browserless_api_key else fetcher
    
    print(f"🔄 Starting TRUE streaming processing: fetch → process → repeat")
    
    # Process each video individually with cancellation checks
    for i, video_url in enumerate(video_urls, 1):
        # Check cancellation before each video
        if task_id:
            task_status = await task_manager.get_task_status(task_id)
            if task_status == TaskStatus.CANCELLED:
                print(f"❌ Task cancelled while processing video {i}/{total_videos}")
                break
        
        video_id = extract_video_id(video_url)
        if not video_id:
            print(f"❌ Invalid video URL: {video_url}")
            continue
        print(f"\n🎯 Processing video {i}/{total_videos}: {video_id}")
        
        # Update progress for this video
        if task_id:
            progress = 10 + (i * 70 // total_videos)  # 10-80% for individual videos
            await task_manager.update_task(task_id, progress=progress, 
                                         message=f"Processing video {i}/{total_videos}: {video_id}")
        
        try:
            # STEP 1: Fetch transcript for this video only
            print(f"📄 Fetching transcript for video {video_id}...")
            
            def _fetch_single_transcript():
                return temp_fetcher.get_transcript(video_url)
            
            transcript = await asyncio.to_thread(_fetch_single_transcript)
            
            # Check cancellation after transcript fetch
            if task_id:
                task_status = await task_manager.get_task_status(task_id)
                if task_status == TaskStatus.CANCELLED:
                    print(f"❌ Task cancelled after fetching transcript for video {video_id}")
                    break
            
            if not transcript:
                print(f"❌ No transcript available for video {video_id}")
                continue
            
            print(f"✅ Transcript fetched for video {video_id}")
            
            # STEP 2: Create video data object
            video_data = VideoData(video_id, playlist_id, userId, api_key)
            video_data.transcript = transcript
            
            # STEP 3: Process translation if needed
            print(f"🔄 Processing translation for video {video_id}...")
            hindi_chars = set('अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह')
            if any(char in hindi_chars for char in transcript[:1000]):
                print(f"Hindi transcript detected for video {video_id}, translating...")
                try:
                    def _translate_sync():
                        return translate_hindi_to_english(transcript, api_key)
                    
                    video_data.transcript = await asyncio.to_thread(_translate_sync)
                    print(f"✅ Translation completed for video {video_id}")
                except Exception as e:
                    print(f"❌ Translation failed for video {video_id}: {e}")
                    # Continue with original transcript
            
            # Check cancellation after translation
            if task_id:
                task_status = await task_manager.get_task_status(task_id)
                if task_status == TaskStatus.CANCELLED:
                    print(f"❌ Task cancelled after translation for video {video_id}")
                    break
            
            # STEP 4: Generate summary
            print(f"🤖 Generating AI summary for video {video_id}...")
            
            # Check if summary already cached
            cached_summary = fetcher._get_cached_summary_from_vector_store(video_id=video_id)
            if cached_summary:
                print(f"📦 Using cached summary for video {video_id}")
                video_data.summary = cached_summary
                video_data.cached = True
            else:
                # Generate new summary
                try:
                    prompt = PromptTemplate(
                        template="""You are an AI assistant that summarizes YouTube videos based on their transcripts.  
                        Treat the user like a bro. keep language semi formal but friendly and chill.

                        Please provide a comprehensive summary with:

                        1. **Overview**: What is this video about in 2-3 sentences?

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
                    
                    # Check cancellation before AI processing
                    if task_id:
                        task_status = await task_manager.get_task_status(task_id)
                        if task_status == TaskStatus.CANCELLED:
                            print(f"❌ Task cancelled before AI processing for video {video_id}")
                            break
                    
                    summary = await asyncio.to_thread(chain.invoke, {"context": video_data.transcript})
                    
                    # Check cancellation after AI processing
                    if task_id:
                        task_status = await task_manager.get_task_status(task_id)
                        if task_status == TaskStatus.CANCELLED:
                            print(f"❌ Task cancelled after AI processing for video {video_id} - not caching")
                            break
                    
                    # Cache summary only if not cancelled
                    fetcher._cache_summary_to_vector_store(summary, video_id=video_id, playlist_id=playlist_id)
                    video_data.summary = summary
                    print(f"✅ AI summary generated and cached for video {video_id}")
                    
                except Exception as e:
                    print(f"❌ Error generating summary for video {video_id}: {e}")
                    video_data.error = f"Error generating summary: {e}"
                    continue
            
            # Add completed video to results
            completed_videos.append(video_data)
            videos_processed += 1
            print(f"🎉 Video {video_id} completed successfully ({videos_processed}/{total_videos})")
            
        except Exception as e:
            print(f"❌ Error processing video {video_id}: {e}")
            continue
    
    print(f"📊 Streaming pipeline complete: {len(completed_videos)}/{total_videos} videos processed")
    return completed_videos


async def generate_final_playlist_summary(completed_videos, playlist_id, api_key, task_id=None):
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

async def summarizePlaylist(playlist_id, userId, api_key, browserless_api_key=None, task_id=None):
    """
    Streaming playlist summarizer that processes videos through concurrent pipeline stages
    Each video flows through: fetch → translate → summarize → collect
    """
    print(f"🚀 Starting streaming playlist summarization: {playlist_id}, task: {task_id}")
    
    # Update task status to running
    if task_id:
        await task_manager.update_task(task_id, status=TaskStatus.RUNNING, progress=5, message="Starting playlist summarization")
    
    try:
        # Check if playlist summary already exists in centralized vector store
        cached_summary = fetcher._get_cached_summary_from_vector_store(playlist_id=playlist_id)
        if cached_summary:
            print("📦 Found cached playlist summary in centralized vector store")
            # Return cached summary (background task will handle WebSocket update)
            return cached_summary
        
        # Check for cancellation before starting pipeline
        if task_id:
            task_status = await task_manager.get_task_status(task_id)
            if task_status == TaskStatus.CANCELLED:
                print(f"Task cancelled: summarize playlist {playlist_id} - before starting pipeline")
                return "Task was cancelled"
        
        # Update progress
        if task_id:
            await task_manager.update_task(task_id, progress=10, message="Starting video processing pipeline")
        
        # Run streaming pipeline
        print("🔄 Starting streaming pipeline...")
        completed_videos = await streaming_playlist_pipeline(playlist_id, userId, api_key, max_concurrent=3, 
                                                           browserless_api_key=browserless_api_key, task_id=task_id)
        
        # Check for cancellation after pipeline
        if task_id:
            task_status = await task_manager.get_task_status(task_id)
            if task_status == TaskStatus.CANCELLED:
                print(f"Task cancelled: summarize playlist {playlist_id} - after pipeline")
                return "Task was cancelled"
        
        # Handle case where no videos were processed successfully
        if not completed_videos:
            if task_id:
                await task_manager.update_task(task_id, status=TaskStatus.FAILED, progress=100, 
                                             message="No videos could be processed", error="No videos could be processed successfully from this playlist.")
            return "No videos could be processed successfully from this playlist."
        
        # Update progress
        if task_id:
            await task_manager.update_task(task_id, progress=85, message="Generating final playlist summary")
        
        # Generate final playlist summary
        print("🔄 Generating final playlist summary...")
        final_summary = await generate_final_playlist_summary(completed_videos, playlist_id, api_key, task_id=task_id)
        
        # Final cancellation check
        if task_id:
            task_status = await task_manager.get_task_status(task_id)
            if task_status == TaskStatus.CANCELLED:
                print(f"Task cancelled: summarize playlist {playlist_id} - after final summary")
                return "Task was cancelled"
        
        # Return final summary (background task will handle WebSocket update)
        print(f"✅ Playlist summarization completed successfully!")
        return final_summary
        
    except Exception as e:
        # Make sure to update task status on error
        if task_id:
            await task_manager.update_task(task_id, status=TaskStatus.FAILED, progress=100, 
                                         message="Playlist summarization failed", error=str(e))
        print(f"❌ Error in summarizePlaylist: {e}")
        raise e
 
