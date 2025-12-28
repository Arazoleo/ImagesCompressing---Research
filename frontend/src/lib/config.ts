/**
 * Configuration constants for the application
 */

export const config = {
  // API Configuration
  API_BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001/api/v1',

  // Frontend Configuration
  MAX_IMAGES_PER_UPLOAD: 10,
  MAX_IMAGE_SIZE: 50 * 1024 * 1024, // 50MB
  ALLOWED_EXTENSIONS: ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'],

  // Processing Configuration
  MAX_CONCURRENT_JOBS: 3,
  POLLING_INTERVAL: 2000, // 2 seconds

  // WebSocket Configuration
  WS_BASE_URL: 'ws://localhost:8001/api/v1',

  // Development flags
  ENABLE_REACT_QUERY_DEVTOOLS: process.env.NEXT_PUBLIC_ENABLE_REACT_QUERY_DEVTOOLS === 'true',
} as const;

export default config;
