// API Configuration
const getApiUrl = (): string => {
  // Use Vercel environment variable in production, localhost in development
  const envApiUrl = (import.meta as any).env?.VITE_API_URL
  if (envApiUrl) {
    return envApiUrl as string
  }
  
  // Default to localhost for local development
  return 'http://localhost:5001'
}

export const API_URL = getApiUrl()

