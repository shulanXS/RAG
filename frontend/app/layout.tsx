import type { Metadata } from 'next'
import '@/styles/globals.css'
import { Providers } from './providers'

export const metadata: Metadata = {
  title: 'Enterprise RAG System',
  description: 'Enterprise-grade RAG application',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>
        <Providers>
          <div className="flex min-h-screen bg-background-secondary">
            {children}
          </div>
        </Providers>
      </body>
    </html>
  )
}
