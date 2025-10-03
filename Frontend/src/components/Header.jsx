import React from 'react';
import { Youtube, Sparkles, Settings } from 'lucide-react';

function Header() {
  const handleSettingsClick = () => {
    // Open options page
    if (chrome.runtime && chrome.runtime.openOptionsPage) {
      chrome.runtime.openOptionsPage();
    }
  };

  return (
    <div className="bg-gradient-to-r from-gray-900 via-gray-850 to-gray-900 border-b border-gray-700/50 backdrop-blur-xl">
      <div className="p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="relative">
              <div className="absolute inset-0 bg-red-600 blur-lg opacity-50 animate-pulse"></div>
              <div className="relative bg-gradient-to-br from-red-600 to-red-700 p-1.5 rounded-lg shadow-lg">
                <Youtube className="w-5 h-5 text-white" />
              </div>
            </div>
            <div>
              <h1 className="text-base font-bold text-white flex items-center">
                Summix
                <Sparkles className="w-3.5 h-3.5 ml-1.5 text-yellow-400 animate-pulse" />
              </h1>
              <p className="text-xs text-gray-400">AI-Powered Insights</p>
            </div>
          </div>
          
          <button
            onClick={handleSettingsClick}
            className="p-1.5 rounded-md bg-gray-800/50 hover:bg-gray-700/50 transition-colors group"
            title="Settings"
          >
            <Settings className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors group-hover:rotate-90 transform duration-300" />
          </button>
        </div>
      </div>
    </div>
  );
}

export default Header;