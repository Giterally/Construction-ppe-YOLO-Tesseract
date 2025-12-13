// API Configuration
const getApiUrl = () => {
  // Use Vercel environment variable in production, localhost in development
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL
  }
  
  // Default to localhost for local development
  return 'http://localhost:5001'
}

export const API_URL = getApiUrl()

