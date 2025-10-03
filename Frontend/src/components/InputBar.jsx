import React, { useState } from 'react';
import { Send, Sparkles, HelpCircle, Lightbulb, MessageSquare } from 'lucide-react';

function InputBar({ onSendMessage, loading }) {
  const [input, setInput] = useState('');
  const [isFocused, setIsFocused] = useState(false);

  const handleSend = () => {
    if (input.trim() && !loading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const quickQuestions = [
    { text: "Main topic?", icon: HelpCircle },
    { text: "Key points", icon: Sparkles },
    { text: "Explain simply", icon: Lightbulb }
  ];

  return (
    <div className="bg-gradient-to-t from-gray-900 via-gray-850 to-gray-800/95 backdrop-blur-xl border-t border-gray-700/50 px-2 pt-2 pb-1">
      <div className="flex gap-2 flex-wrap mb-2 px-1">
        {quickQuestions.map((question, index) => {
          const Icon = question.icon;
          return (
            <button
              key={index}
              onClick={() => onSendMessage(question.text)}
              disabled={loading}
              className="group flex items-center space-x-1.5 px-2.5 py-1 bg-gray-800/80 text-gray-300 rounded-full hover:bg-gray-700/80 disabled:opacity-50 transition-all hover:scale-105 active:scale-95 border border-gray-700/50"
            >
              <Icon className="w-3.5 h-3.5 text-gray-400 group-hover:text-white transition-colors" />
              <span className="text-xs font-medium">{question.text}</span>
            </button>
          );
        })}
      </div>

      <div className="relative">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Ask anything..."
            disabled={loading}
            className="w-full px-3.5 py-2 bg-gray-800/90 text-white rounded-xl focus:outline-none focus:ring-1 focus:ring-red-500/50 disabled:opacity-50 placeholder-gray-500 border border-gray-700/50 transition-all"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="p-2.5 bg-gradient-to-r from-red-600 to-orange-600 text-white rounded-xl hover:from-red-500 hover:to-orange-500 disabled:opacity-50 transition-all hover:scale-105 active:scale-95"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
export default InputBar;