'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import Sidebar from '@/components/layout/Sidebar'

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60 * 1000,
        retry: 1,
      },
    },
  }))

  return (
    <QueryClientProvider client={queryClient}>
      <Sidebar />
      <main className="flex-1 p-6 overflow-y-auto">{children}</main>
    </QueryClientProvider>
  )
}
