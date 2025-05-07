/**
 * API Proxy utility to handle CORS issues
 * This provides a way to make API requests without CORS restrictions
 */

/**
 * Makes an API request using a technique that avoids CORS issues
 * @param url The URL to fetch
 * @param options Fetch options
 * @returns Promise with the response data
 */
export const proxyFetch = async (url: string, options: RequestInit = {}): Promise<any> => {
  try {
    // Log the request for debugging
    console.log(`[PROXY] Fetching: ${url}`);
    
    // Make the request with minimal headers to avoid preflight requests
    const response = await fetch(url, {
      ...options,
      headers: {
        ...(options.headers || {}),
      },
    });

    // Handle non-OK responses
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[PROXY] Error ${response.status}: ${errorText.slice(0, 200)}`);
      throw new Error(`Request failed with status ${response.status}: ${errorText.slice(0, 100)}`);
    }

    // Try to parse as JSON first
    try {
      const data = await response.json();
      console.log(`[PROXY] Success: ${url}`);
      return data;
    } catch (jsonError) {
      // If not JSON, return the text
      const text = await response.text();
      console.log(`[PROXY] Success (text response): ${url}`);
      return { text };
    }
  } catch (error) {
    console.error(`[PROXY] Fetch error for ${url}:`, error);
    throw error;
  }
};

/**
 * Alternative approach using XMLHttpRequest which sometimes works better with CORS
 * @param url The URL to fetch
 * @param token Authentication token
 * @returns Promise with the response data
 */
export const xhrFetch = (url: string, token: string): Promise<any> => {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    
    // Set headers
    if (token) {
      xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    }
    
    xhr.onload = function() {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data);
        } catch (e) {
          resolve({ text: xhr.responseText });
        }
      } else {
        reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
      }
    };
    
    xhr.onerror = function() {
      reject(new Error('Network error'));
    };
    
    xhr.send();
  });
};