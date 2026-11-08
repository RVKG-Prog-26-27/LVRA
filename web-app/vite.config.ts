// needed for Vite to read JSX code (.tsx files)

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    // During development (npm run dev) the AI server runs separately with
    // "python server/app.py" on port 5000. Requests to /api go there.
    proxy: {
      '/api': 'http://127.0.0.1:5000',
    },
  },
});
