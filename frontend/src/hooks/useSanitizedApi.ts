import { useState } from 'react';

/**
 * Custom hook for making API requests with sanitized JSON payloads
 * This ensures that no Infinity values are sent to the backend
 */
export const useSanitizedApi = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Deep sanitize an object to ensure it's safe for JSON serialization
   * Replaces Infinity, -Infinity, NaN with safe values
   */
  const sanitizeForJson = (obj: any): any => {
    // Handle null/undefined
    if (obj === null || obj === undefined) {
      return null;
    }
    
    // Handle primitive values
    if (typeof obj !== 'object') {
      // Handle numbers
      if (typeof obj === 'number') {
        // Check for Infinity, NaN, or very large numbers
        if (!Number.isFinite(obj) || obj === Infinity || obj === -Infinity || isNaN(obj)) {
          // Use a fixed number instead of scientific notation
          return obj > 0 ? 10000000000 : -10000000000;
        }
        return obj;
      }
      
      // Handle strings
      if (typeof obj === 'string') {
        // Check if string represents infinity
        if (obj.toLowerCase().includes('infinity') || obj === 'NaN') {
          return obj.includes('-') ? -1e10 : 1e10;
        }
        return obj;
      }
      
      // Other primitives (boolean, etc.)
      return obj;
    }
    
    // Handle arrays
    if (Array.isArray(obj)) {
      return obj.map(item => sanitizeForJson(item));
    }
    
    // Handle objects
    const result: any = {};
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        result[key] = sanitizeForJson(obj[key]);
      }
    }
    return result;
  };

  /**
   * Helper function to sanitize JSON payload
   */
  const sanitizeJsonPayload = (payload: any): any => {
    // Handle null or undefined
    if (payload === null || payload === undefined) {
      return payload;
    }

    // Handle arrays
    if (Array.isArray(payload)) {
      return payload.map(item => sanitizeJsonPayload(item));
    }

    // Handle objects (but not Date or other special objects)
    if (typeof payload === 'object' && !(payload instanceof Date)) {
      const sanitized: Record<string, any> = {};
      for (const key in payload) {
        if (Object.prototype.hasOwnProperty.call(payload, key)) {
          sanitized[key] = sanitizeJsonPayload(payload[key]);
        }
      }
      return sanitized;
    }

    // Handle special values like Infinity and NaN
    if (typeof payload === 'number') {
      if (!Number.isFinite(payload)) {
        if (payload === Infinity || payload === Number.POSITIVE_INFINITY) {
          return 1e10; // Use a large number instead of Infinity (scientific notation)
        } else if (payload === -Infinity || payload === Number.NEGATIVE_INFINITY) {
          return -1e10; // Use a large negative number instead of -Infinity
        } else if (Number.isNaN(payload)) {
          return 0; // Replace NaN with 0
        }
      }
    }

    // Return other values as is
    return payload;
  };

  /**
   * Make a POST request with a sanitized payload
   */
  const postWithSanitizedPayload = async <T>(
    url: string, 
    payload: any, 
    token?: string
  ): Promise<T> => {
    setLoading(true);
    setError(null);
    
    try {
      // Sanitize the payload
      const sanitizedPayload = sanitizeJsonPayload(payload);
      
      // Double-check for any Infinity values that might have been missed
      const payloadStr = JSON.stringify(sanitizedPayload);
      let finalPayload = sanitizedPayload;
      
      if (payloadStr.includes('Infinity') || payloadStr.includes('NaN')) {
        console.error('WARNING: Payload still contains Infinity or NaN after sanitization!');
        // Replace any remaining Infinity values in the string
        const fixedPayloadStr = payloadStr
          .replace(/Infinity/g, '1e10')
          .replace(/NaN/g, '0');
        
        try {
          // Parse it back to an object
          finalPayload = JSON.parse(fixedPayloadStr);
          console.log('Fixed payload:', finalPayload);
        } catch (e) {
          console.error('Error fixing payload:', e);
        }
      }
      
      // Log the final payload for debugging
      console.log('Request payload:', finalPayload);
      console.log('Request payload (stringified):', JSON.stringify(finalPayload, null, 2));
      console.log('API endpoint:', url);
      
      // Make the request
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      };
      
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
        console.log('Using token:', 'Token exists');
      } else {
        console.log('Using token:', 'No token');
      }
      
      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(finalPayload)
      });
      
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        // Try to get the error details from the response
        const errorText = await response.text();
        console.error('Error response body:', errorText);
        
        // Try to parse as JSON if possible
        try {
          const errorData = JSON.parse(errorText);
          console.error('Parsed error data:', errorData);
        } catch (e) {
          console.error('Could not parse error response as JSON');
        }
        
        throw new Error(`HTTP error ${response.status}: ${errorText}`);
      }
      
      // Parse the response
      const data = await response.json();
      console.log('Response data from API:', data);
      
      setLoading(false);
      return data as T;
    } catch (error: any) {
      setError(error.message || 'An error occurred');
      setLoading(false);
      throw error;
    }
  };

  return {
    loading,
    error,
    postWithSanitizedPayload,
    sanitizeForJson
  };
};
