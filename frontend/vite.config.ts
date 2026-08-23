import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The console talks to the real Python backend (backend/api.py). In dev the
// API runs on :8000 and Vite on :5173, so /api is proxied rather than relying
// on CORS; a production build is served from the same origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
