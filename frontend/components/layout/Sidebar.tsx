'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { MessageSquare, Search, FileText, LayoutDashboard } from 'lucide-react'
import { clsx } from 'clsx'

const navItems = [
  { href: '/chat', label: 'Chat', icon: MessageSquare },
  { href: '/search', label: 'Search', icon: Search },
  { href: '/documents', label: 'Documents', icon: FileText },
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
]

export default function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-60 bg-white border-r border-border flex flex-col shrink-0">
      <div className="p-4">
        <h1 className="text-lg font-bold text-primary">RAG System</h1>
      </div>

      <nav className="flex flex-col gap-1 px-3">
        {navItems.map(({ href, label, icon: Icon }) => {
          const isActive = pathname === href
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-white'
                  : 'text-foreground hover:bg-background-secondary'
              )}
            >
              <Icon size={18} />
              <span>{label}</span>
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}
