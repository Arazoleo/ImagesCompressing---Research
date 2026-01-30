/**
 * Configuration constants for the application
 */

// Detect if running in browser or server
const isBrowser = typeof window !== 'undefined';

// Get API URL based on environment
const getApiUrl = () => {
  // First check environment variable
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  
  // In browser, use relative path or window location
  if (isBrowser) {
    // If running on localhost, use localhost:8001
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://localhost:8001/api/v1';
    }
    // Otherwise, assume backend is on same host but different port
    return `http://${window.location.hostname}:8001/api/v1`;
  }
  
  // Server-side default
  return 'http://backend:8001/api/v1';
};

const getWsUrl = () => {
  if (process.env.NEXT_PUBLIC_WS_URL) {
    return process.env.NEXT_PUBLIC_WS_URL;
  }
  
  if (isBrowser) {
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'ws://localhost:8001/api/v1';
    }
    return `ws://${window.location.hostname}:8001/api/v1`;
  }
  
  return 'ws://backend:8001/api/v1';
};

export const config = {
  // API Configuration
  API_BASE_URL: getApiUrl(),

  // Frontend Configuration
  MAX_IMAGES_PER_UPLOAD: 10,
  MAX_IMAGE_SIZE: 50 * 1024 * 1024, // 50MB
  ALLOWED_EXTENSIONS: ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'],

  // Processing Configuration
  MAX_CONCURRENT_JOBS: 3,
  POLLING_INTERVAL: 2000, // 2 seconds

  // WebSocket Configuration
  WS_BASE_URL: getWsUrl(),

  // Development flags
  ENABLE_REACT_QUERY_DEVTOOLS: process.env.NEXT_PUBLIC_ENABLE_REACT_QUERY_DEVTOOLS === 'true',
} as const;

export default config;
