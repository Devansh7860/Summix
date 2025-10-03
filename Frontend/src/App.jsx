import React, { useState, useEffect, useRef } from 'react';
import MediaInfo from './components/MediaInfo';
import ChatArea from './components/ChatArea';
import InputBar from './components/InputBar';
import Header from './components/Header';
import { summarizeVideo, summarizePlaylist, askQuestion, cancelActiveRequests, getGeminiApiKey, getBrowserlessApiKey, getUserId } from './api';
import webSocketManager from './websocket';

function App() {
  const [mediaInfo, setMediaInfo] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [summarized, setSummarized] = useState(false);
  const [currentTask, setCurrentTask] = useState(null);
  const [taskProgress, setTaskProgress] = useState(0);
  const [taskMessage, setTaskMessage] = useState('');
  const [wsConnected, setWsConnected] = useState(false);
  const previousContentIdRef = useRef(null);

  // Initialize WebSocket connection
  useEffect(() => {
    const initWebSocket = async () => {
      try {
        const userId = await getUserId();
        await webSocketManager.connect(userId);
        
        // Set up task update handler
        const taskUpdateCleanup = webSocketManager.onTaskUpdate((task) => {
          console.log('📊 Task update received:', task);
          
          // Only process task updates for the current content to avoid old task messages
          const currentContentId = mediaInfo?.videoId || mediaInfo?.playlistId;
          if (task.content_id && currentContentId && task.content_id !== currentContentId) {
            console.log('🚫 Ignoring task update for old content:', task.content_id, '(current:', currentContentId, ')');
            return;
          }
          
          // NEVER show cancelled task results in the UI
          if (task.status === 'CANCELLED') {
            console.log('🚫 Ignoring cancelled task update to prevent showing old cancellation messages');
            console.log('🚫 Cancelled task details:', task);
            setLoading(false);
            setCurrentTask(null);
            setTaskProgress(0);
            setTaskMessage('');
            return;
          }
          
          setCurrentTask(task);
          
          // Only update progress for running tasks
          if (task.status === 'RUNNING' || task.status === 'running' || task.status === 'PENDING' || task.status === 'pending') {
            setTaskProgress(task.progress || 0);
            setTaskMessage(task.message || '');
          }
          
          // Handle completed tasks with results
          if (task.status === 'COMPLETED' || task.status === 'completed') {
            if (task.result) {
              console.log('✅ Task completed with result, adding to messages');
              console.log('📋 Result content:', task.result);
              console.log('📋 Result type:', typeof task.result);
              console.log('📋 Result length:', task.result?.length);
              setMessages(prev => [...prev, {
                id: Date.now() + 1,
                type: 'assistant',
                text: task.result,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                isSummary: true
              }]);
              // Clear progress state immediately after successful completion
              setTaskProgress(0);
              setTaskMessage('');
              setLoading(false);
              setCurrentTask(null);
            } else {
              console.log('❌ Task completed but NO RESULT!');
              console.log('📋 Task object:', task);
              setMessages(prev => [...prev, {
                id: Date.now() + 1,
                type: 'assistant',
                text: '⚠️ Summary completed but no content was generated. Please try again.',
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                isError: true
              }]);
            }
          }
          
          // Handle failed tasks
          if (task.status === 'FAILED' || task.status === 'failed') {
            console.log('❌ Task failed:', task.error || task.message);
            setMessages(prev => [...prev, {
              id: Date.now() + 1,
              type: 'assistant',
              text: `⚠️ ${task.error || task.message || 'Summary generation failed. Please try again.'}`,
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
              isError: true
            }]);
            // Clear progress state immediately after failure
            setTaskProgress(0);
            setTaskMessage('');
            setLoading(false);
            setCurrentTask(null);
          }
          
          // Update loading state based on task status
          if (task.status === 'COMPLETED' || task.status === 'FAILED') {
            setLoading(false);
            setCurrentTask(null);
            setTaskProgress(0);
            setTaskMessage('');
          }
        });
        
        // Set up connection status handler
        const connectionCleanup = webSocketManager.onConnectionChange((connected) => {
          setWsConnected(connected);
          console.log('🔌 WebSocket connection status:', connected);
        });
        
        // Cleanup function
        return () => {
          taskUpdateCleanup();
          connectionCleanup();
        };
      } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
      }
    };
    
    initWebSocket();
    
    // Cancel tasks when page/tab closes
    const handleBeforeUnload = () => {
      console.log('🚫 IMMEDIATE CANCELLATION: Page closing!');
      webSocketManager.cancelTasks().catch(console.error);
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    
    // Cleanup on unmount - CANCEL ALL TASKS IMMEDIATELY
    return () => {
      console.log('🚫 IMMEDIATE CANCELLATION: App unmounting!');
      window.removeEventListener('beforeunload', handleBeforeUnload);
      webSocketManager.cancelTasks().catch(console.error);
      webSocketManager.disconnect();
    };
  }, []);

  useEffect(() => {
    const handleMessage = async (request) => {
      if (request.action === 'updateMediaInfo') {
        const response = request.data;
        console.log('UI received state:', response);
        const currentContentId = response?.videoId || response?.playlistId || null;

        if (response.error) {
          if (response.error === 'NOT_YOUTUBE') {
            // Do nothing, preserving the state.
            return;
          }
          // For any other error (like 'NOT_CONTENT_PAGE'), clear everything.
          setMediaInfo(null);
          setMessages([]);
          setSummarized(false);
          previousContentIdRef.current = null;

        } else {
          // We have valid content.
          if (previousContentIdRef.current && currentContentId !== previousContentIdRef.current) {
            // Content changed - IMMEDIATELY cancel all tasks and reset UI
            console.log('Content changed from', previousContentIdRef.current, 'to', currentContentId);
            console.log('� IMMEDIATE CANCELLATION: Content changed!');
            
            // Cancel ALL tasks immediately
            webSocketManager.cancelTasks().catch(console.error);
            
            // IMMEDIATE UI reset - do this first to prevent 2-3 second delay
            setLoading(false);
            setCurrentTask(null);
            setTaskProgress(0);
            setSummarized(false);
            setMessages([]);
            setTaskMessage('');
            setMediaInfo(response); // Update to new content immediately
            
          } else {
            setMediaInfo(response);
          }
          previousContentIdRef.current = currentContentId;
        }
      }
    };
    
    chrome.runtime.onMessage.addListener(handleMessage);
    chrome.runtime.sendMessage({ action: 'requestInitialMediaInfo' });
    
    // Cleanup function to cancel requests when sidepanel closes
    return () => {
      chrome.runtime.onMessage.removeListener(handleMessage);
      console.log('Message listener cleanup');
      // Don't cancel here - let the unmount handler deal with it
    };
  }, []); // REMOVED loading dependency!

  // Add cleanup handlers for when user closes sidepanel
  useEffect(() => {
    const handleBeforeUnload = () => {
      console.log('🚨 Page unloading - cancelling all active requests');
      // Use sync version for beforeunload
      const currentVideoId = mediaInfo?.videoId;
      const currentPlaylistId = mediaInfo?.playlistId;
      if (currentVideoId || currentPlaylistId) {
        webSocketManager.cancelTasks(currentVideoId, currentPlaylistId).catch(console.error);
      }
    };

    // Only use beforeunload - visibilitychange is too aggressive and triggers when switching apps
    window.addEventListener('beforeunload', handleBeforeUnload);

    // Cleanup on component unmount
    return () => {
      console.log('App component unmounting, cancelling any active requests');
      window.removeEventListener('beforeunload', handleBeforeUnload);
      
      // Cancel tasks for current content
      if (mediaInfo?.videoId || mediaInfo?.playlistId) {
        webSocketManager.cancelTasks(mediaInfo?.videoId, mediaInfo?.playlistId).catch(console.error);
      }
    };
  }, [mediaInfo?.videoId, mediaInfo?.playlistId]); // Depend on current content

  // Check if both API keys are configured and valid
  const checkApiKeys = async () => {
    const geminiApiKey = await getGeminiApiKey();
    const browserlessApiKey = await getBrowserlessApiKey();
    
    if (!geminiApiKey || geminiApiKey.trim() === '') {
      return { valid: false, message: 'Gemini API key is not configured. Please set it up in the extension settings.' };
    }
    
    if (!browserlessApiKey || browserlessApiKey.trim() === '') {
      return { valid: false, message: 'Browserless API key is not configured. Please set it up in the extension settings.' };
    }
    
    // API validation is now handled by the token system in api.js
    return { valid: true };
  };
  
  const handleSummarize = async () => {
    if (loading || !mediaInfo) return;

    // Show validating message
    setLoading(true);
    setTaskProgress(0);
    setTaskMessage('Validating API keys...');
    setMessages([{
      id: Date.now(),
      type: 'assistant',
      text: '🔍 Validating API keys...',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }]);

    // Check API keys before proceeding
    const apiKeyCheck = await checkApiKeys();
    if (!apiKeyCheck.valid) {
      setMessages([{
        id: Date.now(),
        type: 'assistant',
        text: `⚠️ ${apiKeyCheck.message}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isError: true
      }]);
      setLoading(false);
      setTaskProgress(0);
      setTaskMessage('');
      return;
    }

    setSummarized(true);

    // If there is a videoId, we always summarize the video.
    // Otherwise, it must be a dedicated playlist page.
    const isVideoSummary = !!mediaInfo.videoId;
    const userMsg = {
      id: Date.now(),
      type: 'user',
      text: isVideoSummary ? '✨ Summarize this video' : '✨ Summarize this playlist',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages([userMsg]);

    try {
      // Make the HTTP request to start the background task
      const response = isVideoSummary
        ? await summarizeVideo(mediaInfo.videoId)
        : await summarizePlaylist(mediaInfo.playlistId);

      if (response?.canceled) {
        console.log('Summary was canceled');
        setLoading(false);
        setTaskProgress(0);
        setTaskMessage('');
        return;
      }

      // The HTTP response now just confirms the task started
      console.log('✅ Summarization task started:', response);
      setTaskMessage('Starting summarization...');
      
      // Results will come via WebSocket - do NOT set loading to false here!
      // The WebSocket task update handler will manage loading state and add results
      
    } catch (error) {
      console.error('Summarization Error:', error);
      
      // Handle aborted requests (when user switches videos) - don't show error message
      if (error.name === 'AbortError') {
        console.log('✅ Request was aborted - not showing error message');
        setLoading(false);
        setTaskProgress(0);
        setTaskMessage('');
        return;
      }
      
      // Handle other types of errors
      console.log('❌ Showing error message for non-abort error');
      setMessages(prev => [...prev, {
        id: Date.now() + 1, 
        type: 'assistant',
        text: '⚠️ Sorry, I was unable to start the summary. Please try again.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), 
        isError: true
      }]);
      setLoading(false);
      setTaskProgress(0);
      setTaskMessage('');
    }
  };
  
  const handleSendMessage = async (input) => {
    if (!input.trim() || loading) return;

    // Show user message and validation message
    const userMsg = {
      id: Date.now(), type: 'user', text: input,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(prev => [...prev, userMsg, {
      id: Date.now() + 1,
      type: 'assistant',
      text: '🔍 Validating API keys...',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }]);
    setLoading(true);

    // Check API keys before proceeding
    const apiKeyCheck = await checkApiKeys();
    if (!apiKeyCheck.valid) {
      setMessages(prev => [...prev.slice(0, -1), {
        id: Date.now() + 2,
        type: 'assistant',
        text: `⚠️ ${apiKeyCheck.message}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        isError: true
      }]);
      setLoading(false);
      return;
    }

    // Remove validation message and continue
    setMessages(prev => prev.slice(0, -1));
    try {
      const response = await askQuestion({
        question: input, videoId: mediaInfo?.videoId, playlistId: mediaInfo?.playlistId
      });
      const answerText = response?.answer;
      console.log('Answer received:', answerText);
      if (typeof answerText === 'string' && answerText.length > 0) {
        setMessages(prev => [...prev, {
          id: Date.now() + 1, type: 'assistant', text: answerText,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }]);
      } else { throw new Error('Invalid answer response from API.'); }
    } catch (error) {
      console.error('Ask Question Error:', error);
      setMessages(prev => [...prev, {
        id: Date.now() + 1, type: 'assistant',
        text: '⚠️ Unable to process your question. Please try again.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), isError: true
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen bg-gradient-to-b from-gray-900 via-gray-900 to-black flex flex-col">
      <Header />
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {mediaInfo && (
          <MediaInfo mediaInfo={mediaInfo} onSummarize={handleSummarize} loading={loading} summarized={summarized} />
        )}
        {!mediaInfo && messages.length === 0 && (
          <div className="m-4">
            <div className="bg-gradient-to-r from-red-900/20 to-red-800/20 rounded-xl p-4">
              <p className="text-red-200 text-sm">Navigate to YouTube content to begin.</p>
            </div>
          </div>
        )}
        <ChatArea 
          messages={messages} 
          loading={loading} 
          taskProgress={taskProgress}
          taskMessage={taskMessage}
          wsConnected={wsConnected}
        />
      </div>
      {summarized && mediaInfo && (
        <InputBar onSendMessage={handleSendMessage} loading={loading} />
      )}
    </div>
  );
}

const style = document.createElement('style');
style.textContent = `
  .custom-scrollbar::-webkit-scrollbar { width: 6px; }
  .custom-scrollbar::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.1); }
  .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(239, 68, 68, 0.5); border-radius: 3px; }
  .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(239, 68, 68, 0.7); }
`;
document.head.appendChild(style);

export default App;





