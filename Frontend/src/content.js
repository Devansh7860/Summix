// content.js - Final version with robust, patient data fetching for all contexts.

// Content script for Summix extension

console.log('Summix Content Script (Final) loaded.');


/**
 * The main function to intelligently get page info, prioritizing video context.
 */
async function getPageInfo() {
  const urlParams = new URLSearchParams(window.location.search);
  const videoId = urlParams.get('v');
  const playlistId = urlParams.get('list');
  console.log(playlistId)

  // Case 1: A video is the primary content. This is the highest priority.
  if (videoId) {
    // We only need the IDs. The background script will fetch the details via API.
    return { type: 'video', videoId, playlistId };
  }

  // Case 2: No video, but it's a dedicated playlist page.
  if (playlistId && window.location.pathname === '/playlist') {
    return { type: 'playlist', playlistId };


  }

  // Case 3: Not a relevant page.
  return null;
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getPageInfo') {
    getPageInfo().then(sendResponse); // passing the data back through sendResponse and return true to indicate async response
    return true; // 
  }
});

