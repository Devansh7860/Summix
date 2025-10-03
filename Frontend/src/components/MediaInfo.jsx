
import React from 'react';
import { Play, List, Sparkles, Loader2, Clock, User } from 'lucide-react';

function MediaInfo({ mediaInfo, onSummarize, loading, summarized }) {
  return (
    <div className="bg-gradient-to-b from-gray-800/95 to-gray-850/95 backdrop-blur-xl border-b border-gray-700/50">
      <div className="p-4">
        <div className="flex space-x-3">
          <div className="relative flex-shrink-0 group">
            {mediaInfo.type === 'video' ? (
              <>
                <img
                  src={mediaInfo.thumbnail}
                  alt={mediaInfo.title}
                  className="w-28 h-[70px] object-cover rounded-lg shadow-lg group-hover:shadow-xl transition-shadow"
                  onError={(e) => {
                    e.target.src = 'https://via.placeholder.com/112x70?text=Video';
                  }}
                />
                <div className="absolute inset-0 bg-black/40 rounded-lg flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <Play className="w-8 h-8 text-white drop-shadow-lg" />
                </div>
              </>
            ) : (
              <div className="w-28 h-[70px] bg-gradient-to-br from-purple-600/20 to-blue-600/20 rounded-lg flex items-center justify-center shadow-lg">
                <img src={mediaInfo.thumbnail} />
              </div>
            )}
            
            <div className={`absolute -top-1 -right-1 px-2 py-0.5 rounded-full text-xs font-bold shadow-lg ${
              mediaInfo.type === 'playlist' 
                ? 'bg-gradient-to-r from-purple-500 to-blue-500 text-white' 
                : 'bg-gradient-to-r from-red-500 to-orange-500 text-white'
            }`}>
              {mediaInfo.type === 'playlist' ? '📚' : '🎬'}
            </div>
          </div>
          
          <div className="flex-1 min-w-0">
            <h2 className="text-sm font-bold text-white line-clamp-2 leading-tight mb-1 hover:text-red-400 transition-colors">
              {mediaInfo.title}
            </h2>
            <div className="flex items-center space-x-3 text-xs">
              <div className="flex items-center space-x-1 text-gray-400">
                <User className="w-3 h-3" />
                <span className="truncate max-w-[120px]">{mediaInfo.author}</span>
              </div>
              {mediaInfo.type === 'playlist' && (
                <div className="flex items-center space-x-1 text-purple-400">
                  <Clock className="w-3 h-3" />
                  <span>Playlist</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {!summarized && (
          <button
            onClick={onSummarize}
            disabled={loading}
            className="w-full mt-3 py-2.5 px-4 bg-gradient-to-r from-red-600 to-orange-600 text-white rounded-xl font-bold shadow-lg hover:shadow-xl hover:from-red-500 hover:to-orange-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition-all transform hover:scale-[1.02] active:scale-[0.98]"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Generating Summary...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Summarize {mediaInfo.type === 'playlist' ? 'Playlist' : 'Video'}</span>
                <span className="ml-1 px-1.5 py-0.5 bg-white/20 rounded-full text-xs">AI</span>
              </>
            )}
          </button>
        )}
        
        {summarized && !loading && (
          <div className="mt-3 flex items-center justify-center space-x-2 text-xs">
            <div className="flex space-x-1">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" style={{ animationDelay: '200ms' }}></span>
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" style={{ animationDelay: '400ms' }}></span>
            </div>
            <span className="text-green-400 font-medium">Summary Generated</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default MediaInfo;