'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { MessageSquare } from 'lucide-react'
import { clsx } from 'clsx'

export default function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-16 bg-white border-r border-border flex flex-col shrink-0 items-center py-4">
      <Link
        href="/chat"
        className={clsx(
          'w-10 h-10 rounded-xl flex items-center justify-center transition-colors',
          pathname === '/chat'
            ? 'bg-primary text-white'
            : 'text-foreground hover:bg-background-secondary'
        )}
      >
        <MessageSquare size={20} />
      </Link>
    </aside>
  )
}
