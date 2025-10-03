import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { Save, Key, Check, AlertCircle, Loader2 } from 'lucide-react';
import { validateGeminiApiKey, validateBrowserlessApiKey, validateAndSaveApiKeys, resetValidationState } from './api';
import './index.css';

function Options() {
  const [settings, setSettings] = useState({
    geminiApiKey: '',
    browserlessApiKey: ''
  });
  const [saved, setSaved] = useState(false);
  const [validating, setValidating] = useState(false);
  const [geminiValidation, setGeminiValidation] = useState(null); // { valid: boolean, message: string }
  const [browserlessValidation, setBrowserlessValidation] = useState(null); // { valid: boolean, message: string }

  // Load saved settings when page opens
  useEffect(() => {
    chrome.storage.sync.get(['settings'], (data) => {
      if (data.settings) {
        setSettings(prev => ({
          ...prev,
          geminiApiKey: data.settings.geminiApiKey || '',
          browserlessApiKey: data.settings.browserlessApiKey || ''
        }));
      }
    });
  }, []);

  // Save settings to Chrome storage
  const handleSave = async () => {
    if (!settings.geminiApiKey.trim()) {
      setGeminiValidation({ valid: false, message: 'Gemini API key is required' });
      return;
    }
    
    if (!settings.browserlessApiKey.trim()) {
      setBrowserlessValidation({ valid: false, message: 'Browserless API key is required' });
      return;
    }

    // Reset previous validation states
    setValidating(true);
    setGeminiValidation(null);
    setBrowserlessValidation(null);
    resetValidationState(); // Reset global validation state
    
    try {
      // Use the new validation and save function
      const result = await validateAndSaveApiKeys(settings.geminiApiKey, settings.browserlessApiKey);
      
      if (result.success) {
        setGeminiValidation({ valid: true, message: 'Gemini API key is valid!' });
        setBrowserlessValidation({ valid: true, message: 'Browserless API key is valid!' });
        setSaved(true);
        
        setTimeout(() => {
          setSaved(false);
          setGeminiValidation(null);
          setBrowserlessValidation(null);
        }, 5000);
        
        console.log('✅ Settings saved and validated:', settings);
      } else {
        // Show specific error for the failed validation
        if (result.error.includes('Gemini')) {
          setGeminiValidation({ valid: false, message: result.error });
          setBrowserlessValidation({ valid: false, message: 'Validation skipped due to Gemini error' });
        } else if (result.error.includes('Browserless')) {
          setGeminiValidation({ valid: true, message: 'Gemini API key is valid!' });
          setBrowserlessValidation({ valid: false, message: result.error });
        } else {
          setGeminiValidation({ valid: false, message: result.error });
          setBrowserlessValidation({ valid: false, message: result.error });
        }
        
        console.error('❌ Validation failed:', result.error);
      }
    } catch (error) {
      setGeminiValidation({ valid: false, message: 'Failed to save settings. Please try again.' });
      setBrowserlessValidation({ valid: false, message: 'Failed to save settings. Please try again.' });
      console.error('❌ Settings save failed:', error);
    } finally {
      setValidating(false);
    }
  };

  // Handle input changes
  const handleChange = (field, value) => {
    setSettings(prev => ({
      ...prev,
      [field]: value
    }));
    // Clear validation when user types
    if (field === 'geminiApiKey') {
      setGeminiValidation(null);
    } else if (field === 'browserlessApiKey') {
      setBrowserlessValidation(null);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 p-8">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2 flex items-center">
            <Key className="w-8 h-8 mr-3 text-red-500" />
            Summix Settings
          </h1>
          <p className="text-gray-400">
            Configure your API keys to get started
          </p>
        </div>

        {/* Settings Form */}
        <div className="bg-gray-800 rounded-lg p-6 space-y-6">
          {/* Gemini API Key */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Gemini API Key *
            </label>
            <input
              type="password"
              value={settings.geminiApiKey}
              onChange={(e) => handleChange('geminiApiKey', e.target.value)}
              placeholder="Enter your Gemini API key"
              className={`w-full px-4 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 placeholder-gray-400 ${
                geminiValidation === null ? 'focus:ring-red-500' :
                geminiValidation.valid ? 'focus:ring-green-500 ring-2 ring-green-500' :
                'focus:ring-red-500 ring-2 ring-red-500'
              }`}
              disabled={validating}
            />
            <p className="text-xs text-gray-500 mt-1">
              Get your API key from <a href="https://aistudio.google.com/app/apikey" target="_blank" className="text-red-400 hover:underline">Google AI Studio</a>
            </p>
            
            {/* Gemini Validation Message */}
            {geminiValidation && (
              <div className={`mt-2 flex items-center space-x-2 text-sm ${
                geminiValidation.valid ? 'text-green-400' : 'text-red-400'
              }`}>
                {geminiValidation.valid ? (
                  <Check className="w-4 h-4" />
                ) : (
                  <AlertCircle className="w-4 h-4" />
                )}
                <span>{geminiValidation.message}</span>
              </div>
            )}
          </div>

          {/* Browserless API Key */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Browserless API Key *
            </label>
            <input
              type="password"
              value={settings.browserlessApiKey}
              onChange={(e) => handleChange('browserlessApiKey', e.target.value)}
              placeholder="Enter your Browserless API key"
              className={`w-full px-4 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 placeholder-gray-400 ${
                browserlessValidation === null ? 'focus:ring-red-500' :
                browserlessValidation.valid ? 'focus:ring-green-500 ring-2 ring-green-500' :
                'focus:ring-red-500 ring-2 ring-red-500'
              }`}
              disabled={validating}
            />
            <p className="text-xs text-gray-500 mt-1">
              Get your API key from <a href="https://www.browserless.io/" target="_blank" className="text-red-400 hover:underline">Browserless.io</a> (1000 free requests/month)
            </p>
            
            {/* Browserless Validation Message */}
            {browserlessValidation && (
              <div className={`mt-2 flex items-center space-x-2 text-sm ${
                browserlessValidation.valid ? 'text-green-400' : 'text-red-400'
              }`}>
                {browserlessValidation.valid ? (
                  <Check className="w-4 h-4" />
                ) : (
                  <AlertCircle className="w-4 h-4" />
                )}
                <span>{browserlessValidation.message}</span>
              </div>
            )}
          </div>

          {/* Save Button */}
          <button
            onClick={handleSave}
            disabled={validating || !settings.geminiApiKey.trim() || !settings.browserlessApiKey.trim()}
            className={`w-full py-3 px-4 rounded-lg font-medium flex items-center justify-center space-x-2 transition-colors ${
              validating || !settings.geminiApiKey.trim() || !settings.browserlessApiKey.trim()
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : saved
                ? 'bg-green-600 text-white hover:bg-green-700'
                : 'bg-red-600 text-white hover:bg-red-700'
            }`}
          >
            {validating ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Validating API Key...</span>
              </>
            ) : saved ? (
              <>
                <Check className="w-5 h-5" />
                <span>Settings Saved!</span>
              </>
            ) : (
              <>
                <Save className="w-5 h-5" />
                <span>Validate & Save Settings</span>
              </>
            )}
          </button>

          {/* Success Message */}
          {saved && (
            <div className="text-green-400 text-sm text-center animate-pulse">
              ✓ Your settings have been saved successfully
            </div>
          )}
        </div>

        {/* Instructions */}
        <div className="mt-8 bg-gray-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-white mb-3">
            How to Use
          </h2>
          <ol className="space-y-2 text-sm text-gray-300">
            <li>1. Get your free Gemini API key from Google AI Studio</li>
            <li>2. Enter your API key above - it will be automatically validated</li>
            <li>3. Click "Validate & Save Settings" to save your configuration</li>
            <li>4. Go to any YouTube video or playlist</li>
            <li>5. Click the extension icon to open the summarizer</li>
            <li>6. Click "Summarize" to generate AI summaries and ask questions</li>
          </ol>
          
          <div className="mt-4 p-3 bg-blue-900/30 rounded-lg border border-blue-500/30">
            <p className="text-sm text-blue-300">
              💡 <strong>Pro tip:</strong> Your API key is validated before saving to ensure it works correctly with the extension.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Mount the Options component
const root = document.getElementById('options-root');
if (root) {
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <Options />
    </React.StrictMode>
  );
}