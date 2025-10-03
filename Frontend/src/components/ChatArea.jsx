import React, { useEffect, useRef } from 'react';
import { User, Sparkles, Loader2, MessageSquare, Zap } from 'lucide-react';

function ChatArea({ messages, loading, taskProgress = 0, taskMessage = '', wsConnected = false }) {
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Show welcome state if no messages
  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="text-center max-w-sm animate-fadeIn">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-red-600/20 to-purple-600/20 blur-3xl"></div>
            <div className="relative bg-gradient-to-br from-gray-800/90 to-gray-900/90 backdrop-blur-xl rounded-2xl p-8 border border-gray-700/50">
              <div className="mb-4">
                <Sparkles className="w-12 h-12 text-red-500 mx-auto animate-pulse" />
              </div>
              <h3 className="text-xl font-bold text-white mb-2 bg-gradient-to-r from-red-400 to-purple-400 bg-clip-text text-transparent">
                Ready to Summarize
              </h3>
              <p className="text-gray-400 text-sm leading-relaxed">
                Click the "Summarize" button above to unlock AI-powered insights from this content
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 p-4 space-y-4">
      {messages.map((message, index) => (
        <MessageBubble 
          key={message.id} 
          message={message} 
          isFirst={index === 0}
          showAnimation={index === messages.length - 1}
        />
      ))}
      
      {loading && (
        <div className="px-4 animate-slideUp">
          <div className="flex items-center space-x-3 mb-2">
            <div className="bg-gradient-to-r from-red-600/20 to-purple-600/20 backdrop-blur-sm rounded-full p-2">
              <Loader2 className="w-5 h-5 text-red-400 animate-spin" />
            </div>
            <div className="flex space-x-1">
              <span className="w-2 h-2 bg-red-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
              <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
              <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
            </div>
            <span className="text-sm text-gray-400">
              {taskMessage || 'AI is analyzing...'}
            </span>
            {!wsConnected && (
              <span className="text-xs text-yellow-500 opacity-75">
                (Connecting...)
              </span>
            )}
          </div>
          
          {/* Progress Bar */}
          <div className="w-full bg-gray-700/50 rounded-full h-2 mb-2 overflow-hidden">
            <div 
              className="bg-gradient-to-r from-red-500 to-purple-500 h-2 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${Math.max(taskProgress, 5)}%` }}
            ></div>
          </div>
          
          <div className="text-xs text-gray-500 text-right">
            {taskProgress > 0 ? `${taskProgress}%` : 'Starting...'}
          </div>
        </div>
      )}
      
      <div ref={messagesEndRef} />
    </div>
  );
}

function MessageBubble({ message, isFirst, showAnimation }) {
  const isUser = message.type === 'user';
  const isSummary = message.isSummary;
  const isError = message.isError;
  
  // Format the message text with markdown-like formatting
  const formatText = (text) => {
    // Replace **text** with bold
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong class="font-bold text-white">$1</strong>');
    // Replace *text* with italic
    text = text.replace(/\*(.*?)\*/g, '<em class="italic">$1</em>');
    // Replace bullet points
    text = text.replace(/^• /gm, '<span class="text-red-400 mr-1">▸</span>');
    // Replace numbered lists
    text = text.replace(/^(\d+)\. /gm, '<span class="text-purple-400 font-semibold">$1.</span> ');
    // Add line breaks for better formatting
    text = text.replace(/\n\n/g, '</p><p class="mt-3">');
    text = text.replace(/\n/g, '<br/>');

    text = text.replace(/^#{3}\s*(.+)$/gm, (m, p1) => `<h2>${p1.trim()}</h2>`);
    
    return `<p>${text}</p>`;
  };
  
  if (isUser) {
    return (
      <div className={`flex justify-end ${showAnimation ? 'animate-slideUp' : ''}`}>
        <div className="flex items-start space-x-2 max-w-[85%] flex-row-reverse space-x-reverse">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0 shadow-lg">
            <User className="w-5 h-5 text-white" />
          </div>
          <div className="group">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-3 rounded-2xl rounded-tr-sm shadow-lg">
              <p className="text-sm font-medium">{message.text}</p>
            </div>
            {message.timestamp && (
              <p className="text-xs text-gray-500 mt-1 text-right opacity-0 group-hover:opacity-100 transition-opacity">
                {message.timestamp}
              </p>
            )}
          </div>
        </div>
      </div>
    );
  }
  
  // AI Assistant message with premium styling
  return (
    <div className={`flex justify-start ${showAnimation ? 'animate-slideUp' : ''}`}>
      <div className="flex items-start space-x-3 max-w-[85%]">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg ${
          isSummary 
            ? 'bg-gradient-to-br from-red-500 to-orange-500 animate-glow' 
            : isError
            ? 'bg-gradient-to-br from-yellow-500 to-red-500'
            : 'bg-gradient-to-br from-gray-600 to-gray-700'
        }`}>
          {isSummary ? (
            <Sparkles className="w-5 h-5 text-white" />
          ) : isError ? (
            <Zap className="w-5 h-5 text-white" />
          ) : (
            <MessageSquare className="w-5 h-5 text-white" />
          )}
        </div>
        
        <div className="group flex-1">
          {isSummary && (
            <div className="flex items-center space-x-2 mb-2">
              <span className="text-xs font-semibold text-transparent bg-gradient-to-r from-red-400 to-orange-400 bg-clip-text">
                AI SUMMARY
              </span>
              <div className="flex space-x-1">
                <span className="w-1 h-1 bg-red-400 rounded-full animate-pulse"></span>
                <span className="w-1 h-1 bg-orange-400 rounded-full animate-pulse" style={{ animationDelay: '200ms' }}></span>
                <span className="w-1 h-1 bg-yellow-400 rounded-full animate-pulse" style={{ animationDelay: '400ms' }}></span>
              </div>
            </div>
          )}
          
          <div className={`relative overflow-hidden rounded-2xl ${
            isSummary 
              ? 'bg-gradient-to-br from-gray-800/95 via-gray-850/95 to-gray-900/95 border border-red-500/20 shadow-2xl' 
              : isError
              ? 'bg-gradient-to-br from-yellow-900/20 to-red-900/20 border border-red-500/30'
              : 'bg-gray-800/90 border border-gray-700/50'
          }`}>
            {isSummary && (
              <div className="absolute inset-0 bg-gradient-to-r from-red-600/5 via-transparent to-orange-600/5 animate-shimmer"></div>
            )}
            
            <div className="relative px-4 py-3">
              <div 
                className={`text-sm leading-relaxed ${
                  isSummary ? 'text-gray-200' : isError ? 'text-red-200' : 'text-gray-300'
                }`}
                dangerouslySetInnerHTML={{ __html: formatText(message.text) }}
              />
            </div>
          </div>
          
          {message.timestamp && (
            <p className="text-xs text-gray-500 mt-1 opacity-0 group-hover:opacity-100 transition-opacity">
              {message.timestamp}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

// Add animations to document
const style = document.createElement('style');
style.textContent = `
  @keyframes slideUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  
  @keyframes glow {
    0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.5); }
    50% { box-shadow: 0 0 30px rgba(239, 68, 68, 0.8); }
  }
  
  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }
  
  .animate-slideUp {
    animation: slideUp 0.4s ease-out;
  }
  
  .animate-fadeIn {
    animation: fadeIn 0.5s ease-out;
  }
  
  .animate-glow {
    animation: glow 2s ease-in-out infinite;
  }
  
  .animate-shimmer {
    animation: shimmer 3s ease-in-out infinite;
  }
`;
if (!document.head.querySelector('#chat-animations')) {
  style.id = 'chat-animations';
  document.head.appendChild(style);
}

export default ChatArea;