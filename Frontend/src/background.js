// background.js - Final version with clear logic and correct error signaling.

async function safeSendMessage(tabId, message) {
  try {
    return await Promise.race([
      chrome.tabs.sendMessage(tabId, message), // chrome.tabs.sendMessage sends message to the content script on a specific tab whose id we pass as an argument
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Timeout")), 3500)
      ),
    ]);
  } catch (error) {
    console.warn(
      `Could not connect to tab ${tabId}. Message: ${error.message}`
    );
    return null;
  }
}

function sendUpdateToUI(data) {
  chrome.runtime.sendMessage({ action: "updateMediaInfo", data: data });
}

async function updateMediaInfoForTab(tabId, tabUrl) {
  const pageInfo = await safeSendMessage(tabId, { action: "getPageInfo" });

  if (pageInfo?.type === "video") {
    // Video Logic: Always use the reliable oEmbed API for video details.
    try {
      const oEmbedUrl = `https://www.youtube.com/oembed?url=http://www.youtube.com/watch?v=${pageInfo.videoId}&format=json`;
      const response = await fetch(oEmbedUrl);
      if (!response.ok) throw new Error("oEmbed fetch failed.");
      const details = await response.json();
      sendUpdateToUI({
        type: "video",
        videoId: pageInfo.videoId,
        playlistId: pageInfo.playlistId,
        title: details.title,
        author: details.author_name,
        thumbnail: `https://img.youtube.com/vi/${pageInfo.videoId}/maxresdefault.jpg`,
      });
    } catch (error) {
      sendUpdateToUI({ error: "Could not fetch video details." });
    }
  } else if (pageInfo?.type === "playlist") {
    try {

      const oEmbedUrl = `https://www.youtube.com/oembed?url=http://www.youtube.com/playlist?list=${pageInfo.playlistId}&format=json`;
      const response = await fetch(oEmbedUrl);
      if (!response.ok) throw new Error("oEmbed fetch failed.");
      const details = await response.json();

      const title = details.title;
      const author = details.author_name
      const thumbnail = details.thumbnail_url || "https://via.placeholder.com/112x70?text=Playlist";
      // const author = playlist?.info?.author?.name || 'Unknown Channel';
      sendUpdateToUI({
        type: "playlist",
        playlistId: pageInfo.playlistId,
        title,
        author,
        thumbnail,
      });
    } catch (error) {
      console.error("Error fetching playlist:", error);
      sendUpdateToUI({ error: "Could not fetch playlist details." });
    }
  } else {
    // Determine why nothing was found and inform the UI.
    if (tabUrl.includes("youtube.com/")) {
      sendUpdateToUI({ error: "NOT_CONTENT_PAGE" });
    } else {
      sendUpdateToUI({ error: "NOT_YOUTUBE" });
    }
  }
}

// Event Listeners
chrome.tabs.onActivated.addListener((activeInfo) => {
  chrome.tabs.get(activeInfo.tabId, (tab) => {
    if (tab) updateMediaInfoForTab(activeInfo.tabId, tab.url);
  });
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.url) {
    updateMediaInfoForTab(tabId, tab.url);
  }
});

chrome.runtime.onMessage.addListener((request) => {
  if (request.action === "requestInitialMediaInfo") {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) updateMediaInfoForTab(tabs[0].id, tabs[0].url);
    });
  }
});

// Standard Extension Setup
chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch((error) => console.error(error));
  chrome.storage.sync.get("userId", (data) => {
    if (!data.userId) {
      const userId = crypto.randomUUID(); // generates unique ID
      chrome.storage.sync.set({ userId });
      console.log("✅ User ID generated:", userId);
    } else {
      console.log("ℹ️ Existing userId:", data.userId);
    }
  });
});
