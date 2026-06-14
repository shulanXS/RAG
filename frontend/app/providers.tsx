'use client'

import Sidebar from '@/components/layout/Sidebar'

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <>
      <Sidebar />
      <main className="flex-1 p-6 overflow-y-auto">{children}</main>
    </>
  )
}
