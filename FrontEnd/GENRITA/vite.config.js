import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/', // ⚠️ اینو حتما بذار ریشه چون قراره با FastAPI روی root سرو بشه
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: ['arsemi.qzz.io'],
  },
  build: {
    outDir: 'dist', // پیش‌فرضه، ولی خوبه صراحتاً بنویسی
  }
})