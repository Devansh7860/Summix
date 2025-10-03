// API functions for backend communication

// Fixed backend URL
const BACKEND_URL = 'https://summix.onrender.com'
// const BACKEND_URL = 'http://127.0.0.1:8000';

// Request tracking
const activeRequests = {
  videoId: null,
  playlistId: null,
  abortController: null
};

// Validation State Management (persisted in Chrome storage)
let validationState = {
  geminiValid: false,
  browserlessValid: false,
  lastValidated: null,
  validationInProgress: false,
  lastValidatedKeys: { geminiApiKey: '', browserlessApiKey: '' } // Track what keys were validated
};

// Load validation state from Chrome storage
const loadValidationState = async () => {
  return new Promise((resolve) => {
    chrome.storage.sync.get(['validationState'], (data) => {
      if (data.validationState) {
        validationState = { ...validationState, ...data.validationState };
        console.log('📥 Loaded validation state from storage:', validationState);
      }
      resolve(validationState);
    });
  });
};

// Save validation state to Chrome storage
const saveValidationState = async () => {
  return new Promise((resolve) => {
    chrome.storage.sync.set({ validationState }, () => {
      console.log('💾 Saved validation state to storage:', validationState);
      resolve();
    });
  });
};

// Reset validation state
const resetValidationState = async () => {
  validationState.geminiValid = false;
  validationState.browserlessValid = false;
  validationState.lastValidated = null;
  validationState.validationInProgress = false;
  // Don't reset lastValidatedKeys here - we need to track them
  await saveValidationState();
};

// Get validation state for debugging
const getValidationState = () => ({
  isValid: validationState.geminiValid && validationState.browserlessValid,
  lastValidated: validationState.lastValidated,
  geminiValid: validationState.geminiValid,
  browserlessValid: validationState.browserlessValid
});

async function getGeminiApiKey() {
  return new Promise((resolve) => {
    chrome.storage.sync.get("settings", (data) => {
      const apiKey = data.settings?.geminiApiKey || '';
      resolve(apiKey);
    });
  });
}

async function getBrowserlessApiKey() {
  return new Promise((resolve) => {
    chrome.storage.sync.get("settings", (data) => {
      const apiKey = data.settings?.browserlessApiKey || '';
      resolve(apiKey);
    });
  });
}

// Export the functions that are needed by other components
export { getGeminiApiKey, getBrowserlessApiKey, resetValidationState, getValidationState, getUserId };

// Simple validation functions for Options.jsx
export const validateGeminiApiKey = async (apiKey) => {
  // Just format check - real validation happens in validateAndSaveApiKeys
  if (!apiKey || !apiKey.startsWith('AIza')) {
    return { valid: false, error: 'Invalid API key format. Gemini API keys should start with "AIza"' };
  }
  return { valid: true, message: 'API key format is valid' };
};

export const validateBrowserlessApiKey = async (apiKey) => {
  // Just format check - real validation happens in validateAndSaveApiKeys
  if (!apiKey || apiKey.length < 20) {
    return { valid: false, error: 'Browserless API key appears to be too short' };
  }
  return { valid: true, message: 'API key format is valid' };
};

async function getValidationToken() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(["validationToken"], (data) => {
      const token = data.validationToken || '';
      resolve(token);
    });
  });
}

// Validation Middleware - validates and saves keys regardless of validation status
export const validateAndSaveApiKeys = async (geminiKey, browserlessKey) => {
  if (validationState.validationInProgress) {
    throw new Error('Validation already in progress');
  }

  validationState.validationInProgress = true;

  try {
    // Always save keys to localStorage first (regardless of validation status)
    console.log('Saving API keys to localStorage...');
    await new Promise((resolve) => {
      chrome.storage.sync.set({ 
        settings: { 
          geminiApiKey: geminiKey.trim(), 
          browserlessApiKey: browserlessKey.trim() 
        }
      }, resolve);
    });

    // Check if keys exist
    if (!geminiKey || geminiKey.trim() === '') {
      validationState.validationInProgress = false;
      await saveValidationState();
      return { success: false, error: 'Gemini API key is required. Please enter your API key.' };
    }

    if (!browserlessKey || browserlessKey.trim() === '') {
      validationState.validationInProgress = false;
      await saveValidationState();
      return { success: false, error: 'Browserless API key is required. Please enter your API key.' };
    }

    // Load current validation state
    await loadValidationState();

    // Check if keys have changed since last validation
    const keysChanged = (
      geminiKey !== validationState.lastValidatedKeys.geminiApiKey ||
      browserlessKey !== validationState.lastValidatedKeys.browserlessApiKey
    );

    // If keys haven't changed and we have valid status, keep previous validation status
    if (!keysChanged && (validationState.geminiValid && validationState.browserlessValid)) {
      validationState.validationInProgress = false;
      await saveValidationState();
      console.log('✅ Keys unchanged, keeping previous validation status');
      return { success: true, message: 'Keys saved. Previous validation status maintained.' };
    }

    // Keys changed or not validated before, validate fresh
    console.log('Keys changed or not validated, validating fresh...');
    // Load current validation state first
    await loadValidationState();
    
    // Don't reset state - just update the validation flags to false
    validationState.geminiValid = false;
    validationState.browserlessValid = false;
    validationState.validationInProgress = true;

    // Validate both keys with single backend call
    console.log('Validating both API keys with backend...');
    try {
      const response = await fetch(`${BACKEND_URL}/validate-both-keys`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-Gemini-API-Key': geminiKey,
          'X-Browserless-API-Key': browserlessKey
        },
        body: JSON.stringify({ test: true }),
        signal: AbortSignal.timeout(35000) // 35 second timeout
      });

      const result = await response.json();
      
      if (!result.valid) {
        validationState.lastValidatedKeys = { geminiApiKey: geminiKey, browserlessApiKey: browserlessKey };
        validationState.validationInProgress = false;
        await saveValidationState();
        return { success: false, error: result.error };
      }

      // Store validation token in Chrome storage
      await new Promise((resolve) => {
        chrome.storage.sync.set({ 
          validationToken: result.validation_token
        }, resolve);
      });

      validationState.geminiValid = true;
      validationState.browserlessValid = true;
      
    } catch (error) {
      validationState.lastValidatedKeys = { geminiApiKey: geminiKey, browserlessApiKey: browserlessKey };
      validationState.validationInProgress = false;
      await saveValidationState();
      if (error.name === 'TimeoutError') {
        return { success: false, error: 'Validation timeout. Please check your internet connection.' };
      }
      return { success: false, error: `Validation failed: ${error.message}` };
    }

    // Both keys are valid - save state
    validationState.lastValidated = new Date().toISOString();
    validationState.lastValidatedKeys = { geminiApiKey: geminiKey, browserlessApiKey: browserlessKey };
    validationState.validationInProgress = false;
    await saveValidationState();
    
    console.log('✅ Both API keys validated successfully');
    return { success: true, message: 'Both API keys are valid and ready to use!' };

  } catch (error) {
    validationState.validationInProgress = false;
    await saveValidationState();
    return { success: false, error: error.message };
  }
};

// Middleware to check validation before operations
const requireValidation = async () => {
  // Load validation state from storage first
  await loadValidationState();
  
  // Get current keys
  const currentGeminiKey = await getGeminiApiKey();
  const currentBrowserlessKey = await getBrowserlessApiKey();
  
  console.log('🔍 requireValidation() - Current keys exist:', {
    gemini: !!currentGeminiKey,
    browserless: !!currentBrowserlessKey
  });
  
  console.log('🔍 requireValidation() - Validation state:', {
    geminiValid: validationState.geminiValid,
    browserlessValid: validationState.browserlessValid,
    lastValidatedKeys: validationState.lastValidatedKeys
  });
  
  // Check if keys have changed since last validation
  const keysChanged = (
    currentGeminiKey !== validationState.lastValidatedKeys.geminiApiKey ||
    currentBrowserlessKey !== validationState.lastValidatedKeys.browserlessApiKey
  );
  
  console.log('🔍 requireValidation() - Keys changed:', keysChanged);
  
  // If keys changed, validation status is invalid
  if (keysChanged) {
    throw new Error('API keys have changed. Please validate your keys in the extension settings before using the app.');
  }
  
  // Check current validation state
  const state = getValidationState();
  console.log('🔍 requireValidation() - State is valid:', state.isValid);
  
  if (!state.isValid) {
    throw new Error('API keys are not validated. Please configure and validate your API keys in the extension settings first.');
  }
  
  return true;
};

async function getUserId() {
  return new Promise((resolve) => {
    chrome.storage.sync.get("userId", (data) => {
      resolve(data.userId);
      console.log("Fetched userId:", data.userId);
    });
  });
}

// Cancel any active requests
export const cancelActiveRequests = async () => {
  if (activeRequests.abortController && !activeRequests.abortController.signal.aborted) {
    console.log('Canceling active requests for:', 
      activeRequests.videoId || activeRequests.playlistId);
    
    // Store current request info before clearing
    const videoId = activeRequests.videoId;
    const playlistId = activeRequests.playlistId;
    
    // Abort the fetch request with a reason
    activeRequests.abortController.abort('User cancelled request');
    
    // Tell the backend to cancel processing with a fresh request (no abort controller)
    try {
      const userId = await getUserId();
      const geminiApiKey = await getGeminiApiKey();
      
      // Use a fresh fetch request without any abort controller
      await fetch(`${BACKEND_URL}/cancel`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'X-Gemini-API-Key': geminiApiKey
        },
        body: JSON.stringify({ 
          userId,
          videoId,
          playlistId
        })
      });
      console.log('Successfully notified backend of cancellation');
    } catch (error) {
      console.log('Error canceling on backend, but frontend abort was successful:', error);
    }
    
    // Reset tracking AFTER backend notification
    activeRequests.videoId = null;
    activeRequests.playlistId = null;
    activeRequests.abortController = null;
    
  } else if (activeRequests.abortController) {
    console.log('Request already aborted, just clearing tracking');
    activeRequests.videoId = null;
    activeRequests.playlistId = null;
    activeRequests.abortController = null;
  }
};

// Summarize a YouTube video
export const summarizeVideo = async (videoId) => {
  console.log('Summarizing video:', videoId);
  
  // Check validation state first
  try {
    await requireValidation();
  } catch (error) {
    console.error('Validation failed:', error.message);
    throw error; // Re-throw so the UI can handle it
  }
  
  // Note: No longer auto-cancelling previous requests to avoid race conditions
  // Previous requests will be cancelled by the backend when creating new tasks
  
  // Set up new request tracking
  activeRequests.videoId = videoId;
  activeRequests.abortController = new AbortController();
  
  const userId = await getUserId();
  const geminiApiKey = await getGeminiApiKey();
  const browserlessApiKey = await getBrowserlessApiKey();
  const validationToken = await getValidationToken();
  
  console.log('🔍 summarizeVideo - API call data:', {
    userId,
    geminiApiKey: geminiApiKey ? `${geminiApiKey.slice(0, 10)}...` : 'missing',
    browserlessApiKey: browserlessApiKey ? `${browserlessApiKey.slice(0, 10)}...` : 'missing',
    validationToken: validationToken ? `${validationToken.slice(0, 20)}...` : 'missing'
  });
  
  try {
    console.log('🚀 Making summarize request to backend...');
    
    // Double-check that we have a valid AbortController before using it
    const controller = activeRequests.abortController;
    if (!controller || controller.signal.aborted) {
      console.log('⚠️ AbortController is invalid, creating a new one');
      activeRequests.abortController = new AbortController();
    }
    
    const response = await fetch(`${BACKEND_URL}/summarize/video`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'X-Gemini-API-Key': geminiApiKey,
        'X-Browserless-API-Key': browserlessApiKey,
        'X-Validation-Token': validationToken
      },
      body: JSON.stringify({ video_id: videoId, userId }),
      signal: activeRequests.abortController.signal
    });
    
    console.log('✅ Response received:', response.status, response.statusText);
    
    const result = await response.json();
    console.log('📄 Response data:', result);
    
    // Clear tracking only after successful completion
    activeRequests.videoId = null;
    activeRequests.abortController = null;
    
    return result;
  } catch (error) {
    console.error('❌ Summarize request failed:', error);
    
    // Clear tracking on any error
    activeRequests.videoId = null;
    activeRequests.abortController = null;
    
    if (error.name === 'AbortError') {
      console.log('Request was canceled');
      return { summary: null, canceled: true };
    }
    throw error;
  }
};

// Summarize a YouTube playlist
export const summarizePlaylist = async (playlistId) => {
  console.log('Summarizing playlist:', playlistId);
  
  // Check validation state first
  await requireValidation();
  
  // Note: No longer auto-cancelling previous requests to avoid race conditions
  // Previous requests will be cancelled by the backend when creating new tasks
  
  // Set up new request tracking
  activeRequests.playlistId = playlistId;
  activeRequests.abortController = new AbortController();
  
  const userId = await getUserId();
  const geminiApiKey = await getGeminiApiKey();
  const browserlessApiKey = await getBrowserlessApiKey();
  const validationToken = await getValidationToken();
  
  try {
    // Double-check that we have a valid AbortController before using it
    const controller = activeRequests.abortController;
    if (!controller || controller.signal.aborted) {
      console.log('⚠️ AbortController is invalid, creating a new one');
      activeRequests.abortController = new AbortController();
    }
    
    const response = await fetch(`${BACKEND_URL}/summarize/playlist`, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'X-Gemini-API-Key': geminiApiKey,
        'X-Browserless-API-Key': browserlessApiKey,
        'X-Validation-Token': validationToken
      },
      body: JSON.stringify({ playlist_id: playlistId, userId }),
      signal: activeRequests.abortController.signal
    });
    
    const result = await response.json();
    
    // Clear tracking only after successful completion
    activeRequests.playlistId = null;
    activeRequests.abortController = null;
    
    return result;
  } catch (error) {
    // Clear tracking on any error
    activeRequests.playlistId = null;
    activeRequests.abortController = null;
    
    if (error.name === 'AbortError') {
      console.log('Request was canceled');
      return { summary: null, canceled: true };
    }
    throw error;
  }
};

// Answer questions about the content
export const askQuestion = async ({ question, videoId, playlistId }) => {
  console.log('Asking question:', question);
  
  // Check validation state first
  await requireValidation();
  
  const userId = await getUserId();
  const geminiApiKey = await getGeminiApiKey();
  const browserlessApiKey = await getBrowserlessApiKey();
  const validationToken = await getValidationToken();
  
  const response = await fetch(`${BACKEND_URL}/ask`, {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      'X-Gemini-API-Key': geminiApiKey,
      'X-Browserless-API-Key': browserlessApiKey,
      'X-Validation-Token': validationToken
    },
    body: JSON.stringify({ 
      video_id: videoId, 
      userId, 
      playlist_id: playlistId, 
      question 
    })
  });
  
  const res = await response.json();
  return { answer: res.answer };
};