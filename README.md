# 🎯 Summix - AI-Powered YouTube Summarizer

**Summix** is a powerful Chrome extension that uses AI to summarize YouTube videos and playlists in real-time. Get instant insights, key takeaways, and comprehensive summaries without watching entire videos.

## ✨ Features

### 🎥 **Video Summarization**
- **Instant Summaries**: Get comprehensive AI-generated summaries of any YouTube video
- **Smart Caching**: Previously summarized videos load instantly
- **Real-time Progress**: Watch the summarization progress with live updates
- **Interactive Q&A**: Ask questions about the video content and get AI-powered answers

### 📋 **Playlist Summarization**
- **Bulk Processing**: Summarize entire playlists with one click
- **Smart Resume**: Interrupted summaries resume from where they left off
- **Individual & Combined**: Get both individual video summaries and a comprehensive playlist overview
- **Streaming Pipeline**: Videos are processed in real-time as they're completed

### 🚀 **Advanced Features**
- **Multi-language Support**: Automatic Hindi to English translation
- **WebSocket Real-time**: Live progress updates and task management
- **Cancellation Support**: Stop processing anytime with immediate cancellation
- **Chrome Integration**: Seamless sidepanel experience within YouTube
- **Persistent Storage**: All summaries are cached for instant access

## 🛠️ Tech Stack

### **Frontend (Chrome Extension)**
- **React 19** with hooks and modern patterns
- **Tailwind CSS** for responsive, modern UI
- **Vite** for fast development and building
- **WebSocket Client** for real-time communication
- **Chrome Extension APIs** for deep browser integration

### **Backend (AI Processing)**
- **FastAPI** for high-performance async API
- **WebSocket** for real-time bidirectional communication
- **Google Gemini** for AI summarization and question answering
- **LangChain** for prompt engineering and AI workflow
- **Vector Store** for intelligent caching and retrieval
- **Browserless API** for reliable transcript extraction (web scraping)

> **Note**: We use Browserless API for transcript extraction instead of YouTube Transcript API to avoid IP blocking issues in production. The YouTube Transcript API can lead to server IP bans when used at scale, so we rely on web scraping through Browserless for reliable, consistent access to video transcripts.

### **Infrastructure**
- **Render.com** for backend hosting
- **Browserless API** for reliable web scraping
- **Pinecone Vector Store** for caching transcripts and summaries

## 🚀 Quick Start

### **Prerequisites**
- Node.js 18+ and npm
- Python 3.8+
- Chrome browser
- API Keys (Gemini AI, Browserless)

### **1. Setup Backend**

```bash
cd Backend
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your_gemini_api_key"
export BROWSERLESS_API_KEY="your_browserless_api_key"

# Run development server
uvicorn main:app --reload
```

### **2. Setup Frontend**

```bash
cd Frontend
npm install
npm run build

# Load extension in Chrome:
# 1. Open chrome://extensions/
# 2. Enable "Developer mode"
# 3. Click "Load unpacked" and select the dist/ folder
```

### **3. Configure Extension**
1. Click the Summix extension icon
2. Go to Settings and enter your API keys
3. Navigate to any YouTube video or playlist
4. Click "Summarize" and enjoy! 🎉

## 📱 Usage

### **Video Summarization**
1. Navigate to any YouTube video
2. Open the Summix sidepanel
3. Click "Summarize Video"
4. Watch real-time progress as AI processes the content
5. Get comprehensive summary with key concepts and takeaways

### **Playlist Summarization**
1. Open any YouTube playlist
2. Click "Summarize Playlist"
3. Monitor progress as each video is processed
4. Get individual video summaries + combined playlist overview
5. Resume anytime if interrupted

### **Interactive Q&A**
1. After summarization, use the chat feature
2. Ask specific questions about the content
3. Get AI-powered answers based on the video/playlist content

## 🏗️ Architecture

### **Smart Caching System**
- **Video Level**: Individual video summaries cached permanently
- **Playlist Level**: Complete playlist summaries with smart resume
- **Vector Store**: Semantic search for relevant cached content

### **Real-time Processing**
- **WebSocket Connection**: Bidirectional real-time communication
- **Task Management**: Comprehensive task lifecycle with cancellation
- **Progress Updates**: Live progress bars and status messages
- **Error Handling**: Graceful failure recovery and user notifications

### **AI Pipeline**
```
YouTube URL → Transcript Extraction → Language Detection → 
Translation (if needed) → AI Summarization → Caching → 
Real-time Delivery → Interactive Q&A
```

## 🔧 Development

### **Development Commands**

**Backend:**
```bash
# Development with auto-reload
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
# Development build with watch
npm run dev

# Production build
npm run build

# Linting
npm run lint
```

## 🌟 Key Features Deep Dive

### **Smart Resume Technology**
Summix remembers your progress! If you cancel a playlist summarization halfway through, when you restart it:
- ✅ Loads already processed video summaries from cache
- ✅ Continues from where you left off
- ✅ Combines cached + new summaries for complete playlist overview

### **Real-time WebSocket Communication**
- **Live Progress**: See exactly which video is being processed
- **Instant Cancellation**: Stop processing immediately without lag
- **Status Updates**: Real-time feedback on every operation
- **Connection Management**: Automatic reconnection and error handling

### **AI-Powered Intelligence**
- **Context-Aware**: Understands video content and provides relevant summaries
- **Question Answering**: Ask specific questions and get accurate answers
- **Multi-language**: Automatic detection and translation of Hindi content
- **Structured Output**: Organized summaries with overview, key concepts, and takeaways

## 🚀 Deployment

### **Backend Deployment (Render.com)**
1. Connect your repository to Render
2. Set environment variables (API keys)
3. Deploy with single worker configuration for WebSocket support

### **Extension Distribution**
1. Build production version: `npm run build`
2. Package the `dist/` folder
3. Submit to Chrome Web Store

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and feel free to submit issues and enhancement requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini for powerful AI capabilities
- YouTube Transcript API for reliable content extraction
- The open-source community for excellent tools and libraries

---

**Built with ❤️ by the Summix Team**

*Transform your YouTube experience with AI-powered insights!* 🎯
