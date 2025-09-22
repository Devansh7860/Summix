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

def fetch_transcript(video_id, api_key):
    print(f"Fetching transcript for video ID: {video_id}")
    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        print("Using get_transcript to fetch the transcript...")
        transcript = get_transcript(video_url)
        
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
task_lock = threading.Lock()

def cancel_task(userId, contentId, is_playlist=False):
    """Cancel an active task for a user"""
    task_key = f"{userId}_{contentId}_{is_playlist}"
    
    with task_lock:
        if task_key in active_tasks:
            active_tasks[task_key] = False
            print(f"Cancellation requested for task: {task_key}")
            return True
        return False

def is_task_cancelled(userId, contentId, is_playlist=False):
    """Check if a task has been cancelled"""
    task_key = f"{userId}_{contentId}_{is_playlist}"
    
    with task_lock:
        # If task isn't being tracked or is marked False, it's cancelled
        return task_key not in active_tasks or not active_tasks[task_key]

def register_task(userId, contentId, is_playlist=False):
    """Register a new active task"""
    task_key = f"{userId}_{contentId}_{is_playlist}"
    
    with task_lock:
        active_tasks[task_key] = True
        print(f"Registered new task: {task_key}")

def unregister_task(userId, contentId, is_playlist=False):
    """Remove a task from tracking"""
    task_key = f"{userId}_{contentId}_{is_playlist}"
    
    with task_lock:
        if task_key in active_tasks:
            del active_tasks[task_key]
            print(f"Unregistered task: {task_key}")

# main function to summarize video
def summarize(video_id, userId, api_key):
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

        transcript = fetch_transcript(video_id, api_key)
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



async def fetch_transcript_async(video_id, api_key):
    try:
        def _fetch_transcript_sync(vid):
            print(f"Async: Fetching transcript for video ID: {vid}")
            video_url = f"https://www.youtube.com/watch?v={vid}"
            
            # Use get_transcript to fetch the transcript
            transcript = get_transcript(video_url)
            if not transcript:
                print(f"No transcript available for video ID: {vid}")
                return None
                
            # Simple detection for Hindi script
            hindi_chars = set('अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह')
            if any(char in hindi_chars for char in transcript[:1000]):
                print(f"Async: Hindi transcript detected for video ID: {vid}, translating...")
                english_transcript = translate_hindi_to_english(transcript, api_key)
                return english_transcript
                
            return transcript
        
        transcript = await asyncio.to_thread(_fetch_transcript_sync, video_id)
        return video_id, transcript

    except Exception as e:
        print(f"Error fetching transcript for {video_id}: {e}")
        return video_id, None

async def summarizePlaylist(playlist_id, userId, api_key):
    print("Summarizing playlist:", playlist_id)
    
    # Register this task
    register_task(userId, playlist_id, is_playlist=True)
    
    try:
        # Check if playlist summary already exists in centralized vector store
        cached_summary = fetcher._get_cached_summary_from_vector_store(playlist_id=playlist_id)
        if cached_summary:
            print("📦 Found cached playlist summary in centralized vector store")
            unregister_task(userId, playlist_id, is_playlist=True)
            return cached_summary
        
        playlist_url = "https://www.youtube.com/playlist?list=" + playlist_id
        playlist = Playlist(playlist_url)
        video_urls = [f"https://www.youtube.com/watch?v={video.video_id}" for video in playlist.videos]

        # Check for cancellation after fetching playlist info
        if is_task_cancelled(userId, playlist_id, is_playlist=True):
            print(f"Task cancelled: summarize playlist {playlist_id} - after fetching video IDs")
            unregister_task(userId, playlist_id, is_playlist=True)
            return "Task was cancelled"

        print(f"🚀 Using optimized playlist transcript fetching for {len(video_urls)} videos...")
        
        # Use the optimized playlist fetching method (already using centralized caching)
        def _fetch_playlist_sync():
            return fetcher.get_playlist_transcripts(video_urls)
        
        # Run the optimized playlist fetching in a thread
        transcript_results = await asyncio.to_thread(_fetch_playlist_sync)
        
        # Check for cancellation after transcripts fetched
        if is_task_cancelled(userId, playlist_id, is_playlist=True):
            print(f"Task cancelled: summarize playlist {playlist_id} - after fetching transcripts")
            unregister_task(userId, playlist_id, is_playlist=True)
            return "Task was cancelled"
        
        # Process results and handle translations
        video_transcripts = []
        for video_url, transcript in transcript_results.items():
            if transcript:
                video_id = video_url.split('v=')[1] if 'v=' in video_url else video_url
                
                # Simple detection for Hindi script
                hindi_chars = set('अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह')
                if any(char in hindi_chars for char in transcript[:1000]):
                    print(f"Hindi transcript detected for video ID: {video_id}, translating...")
                    transcript = translate_hindi_to_english(transcript, api_key)
                
                video_transcripts.append((video_id, transcript))
        
        if not video_transcripts:
            unregister_task(userId, playlist_id, is_playlist=True)
            return "No transcripts available for any videos in this playlist."

        # Generate summaries concurrently with limit
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent operations
        model = create_model(api_key)
        chain = prompt | model | parser
        
        async def generate_summary_with_limit(vid, transcript):
            # Check cancellation before starting each summary
            if is_task_cancelled(userId, playlist_id, is_playlist=True):
                return f"Cancelled summary for video {vid}"
                
            async with semaphore:
                try:
                    # Check if summary already cached in centralized store
                    cached_summary = fetcher._get_cached_summary_from_vector_store(video_id=vid)
                    if cached_summary:
                        print(f"📦 Using cached summary for video {vid}")
                        return cached_summary
                    
                    print(f"Generating summary for video ID: {vid}")
                    summary = await asyncio.to_thread(chain.invoke, {"context": transcript})
                    
                    # Check cancellation after generating summary
                    if is_task_cancelled(userId, playlist_id, is_playlist=True):
                        return f"Cancelled after summary for video {vid}"
                    
                    # Cache summary to centralized vector store
                    fetcher._cache_summary_to_vector_store(summary, video_id=vid, playlist_id=playlist_id)
                    return summary
                except Exception as e:
                    print(f"Error generating summary for {vid}: {e}")
                    return f"Error generating summary for video {vid}"

        summary_tasks = [generate_summary_with_limit(vid, transcript) for vid, transcript in video_transcripts]
        summaries = await asyncio.gather(*summary_tasks, return_exceptions=True)
        
        # Check for cancellation after all summaries generated
        if is_task_cancelled(userId, playlist_id, is_playlist=True):
            print(f"Task cancelled: summarize playlist {playlist_id} - after generating summaries")
            unregister_task(userId, playlist_id, is_playlist=True)
            return "Task was cancelled"
        
        # Filter out exceptions and cancellations
        valid_summaries = [s for s in summaries if not isinstance(s, Exception) and not s.startswith("Cancelled")]
        
        combined_summaries = "\n\n".join(
            [f"Video {i+1}: {summary}" for i, summary in enumerate(valid_summaries)]
        )
        
        # Check for cancellation before final summary
        if is_task_cancelled(userId, playlist_id, is_playlist=True):
            print(f"Task cancelled: summarize playlist {playlist_id} - before final summary")
            unregister_task(userId, playlist_id, is_playlist=True)
            return "Task was cancelled"
            
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

        final_model = create_model(api_key)
        final_chain = prompt2 | final_model | parser
        final_summary = await asyncio.to_thread(final_chain.invoke, {"combined_summaries": combined_summaries})
        
        # Cache playlist summary to centralized vector store
        print("💾 Caching playlist summary to centralized vector store...")
        fetcher._cache_summary_to_vector_store(final_summary, playlist_id=playlist_id)
        
        # Unregister completed task
        unregister_task(userId, playlist_id, is_playlist=True)
        return final_summary
        
    except Exception as e:
        # Make sure to unregister on error
        unregister_task(userId, playlist_id, is_playlist=True)
        print(f"Error in summarizePlaylist: {e}")
        raise e
 
