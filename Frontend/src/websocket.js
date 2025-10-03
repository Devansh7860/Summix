// WebSocket connection manager for real-time task updates

// const WEBSOCKET_URL = 'ws://127.0.0.1:8000';
const WEBSOCKET_URL = 'wss://summix.onrender.com';
const BACKEND_URL = 'https://summix.onrender.com';

class WebSocketManager {
  constructor() {
    this.ws = null;
    this.userId = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000; // Start with 1 second
    this.taskUpdateCallbacks = new Set();
    this.connectionCallbacks = new Set();
    this.lastCancelTime = 0;
    this.cancelDebounceMs = 1000; // Prevent rapid cancellations
  }

  async connect(userId) {
    if (this.ws && this.isConnected && this.userId === userId) {
      console.log('WebSocket already connected for user:', userId);
      return;
    }

    this.userId = userId;
    const wsUrl = `${WEBSOCKET_URL}/ws/${userId}`;
    
    try {
      console.log('🔗 Connecting to WebSocket:', wsUrl);
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('✅ WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        
        // Notify connection callbacks
        this.connectionCallbacks.forEach(callback => {
          try {
            callback(true);
          } catch (error) {
            console.error('Error in connection callback:', error);
          }
        });

        // Send ping to keep connection alive
        this.sendPing();
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('📨 WebSocket message received:', message);
          
          if (message.type === 'task_update') {
            // Notify all task update callbacks
            this.taskUpdateCallbacks.forEach(callback => {
              try {
                callback(message.task);
              } catch (error) {
                console.error('Error in task update callback:', error);
              }
            });
          } else if (message.type === 'pong') {
            // Handle pong response
            console.log('🏓 Received pong from server');
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('🔌 WebSocket disconnected:', event.code, event.reason);
        this.isConnected = false;
        
        // Notify connection callbacks
        this.connectionCallbacks.forEach(callback => {
          try {
            callback(false);
          } catch (error) {
            console.error('Error in connection callback:', error);
          }
        });

        // Attempt to reconnect if not manually closed
        if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    this.reconnectAttempts++;
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);
    
    console.log(`🔄 Scheduling reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      if (this.userId && this.reconnectAttempts <= this.maxReconnectAttempts) {
        this.connect(this.userId);
      }
    }, delay);
  }

  sendPing() {
    if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify({ type: 'ping' }));
        console.log('🏓 Sent ping to server');
        
        // Schedule next ping in 30 seconds
        setTimeout(() => this.sendPing(), 30000);
      } catch (error) {
        console.error('Error sending ping:', error);
      }
    }
  }

  disconnect() {
    if (this.ws) {
      console.log('🔌 Manually disconnecting WebSocket');
      this.ws.close(1000, 'Manual disconnect');
      this.ws = null;
    }
    this.isConnected = false;
    this.userId = null;
    this.reconnectAttempts = this.maxReconnectAttempts; // Prevent auto-reconnect
  }

  // Add callback for task updates
  onTaskUpdate(callback) {
    this.taskUpdateCallbacks.add(callback);
    return () => this.taskUpdateCallbacks.delete(callback); // Return cleanup function
  }

  // Add callback for connection status changes
  onConnectionChange(callback) {
    this.connectionCallbacks.add(callback);
    return () => this.connectionCallbacks.delete(callback); // Return cleanup function
  }

  // Send cancellation request through WebSocket (if needed) or fallback to HTTP
  async cancelTasks(videoId = null, playlistId = null) {
    try {
      const { getUserId } = await import('./api.js');
      const userId = await getUserId();
      
      console.log('🚫 Cancelling tasks:', { userId, videoId, playlistId });
      
      const response = await fetch(`${BACKEND_URL}/cancel`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          userId,
          videoId: videoId || undefined,
          playlistId: playlistId || undefined
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('✅ Cancellation result:', result);
      return result;
    } catch (error) {
      console.error('❌ Error cancelling tasks:', error);
      return { success: false, error: error.message };
    }
  }
}

// Create singleton instance
const webSocketManager = new WebSocketManager();

export default webSocketManager;