'use client'

import { useQuery } from '@tanstack/react-query'
import { documentApi } from '@/lib/api'
import { FileText, MessageSquare, Zap } from 'lucide-react'
import { clsx } from 'clsx'

export default function DashboardPage() {
  const { data } = useQuery({
    queryKey: ['documents'],
    queryFn: documentApi.list,
  })

  const stats = [
    {
      label: 'Documents',
      value: data?.total ?? 0,
      icon: FileText,
      color: 'text-primary',
      bg: 'bg-indigo-50',
    },
    {
      label: 'Queries Today',
      value: 0,
      icon: MessageSquare,
      color: 'text-emerald-500',
      bg: 'bg-emerald-50',
    },
    {
      label: 'Cache Hit Rate',
      value: '0%',
      icon: Zap,
      color: 'text-amber-500',
      bg: 'bg-amber-50',
    },
  ]

  return (
    <div>
      <h2 className="text-xl font-semibold mb-6">Dashboard</h2>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {stats.map(({ label, value, icon: Icon, color, bg }) => (
          <div
            key={label}
            className="bg-white border border-border rounded-xl p-6"
          >
            <div className={clsx('w-12 h-12 rounded-lg flex items-center justify-center mb-4', bg)}>
              <Icon size={24} className={color} />
            </div>
            <p className="text-3xl font-bold mb-1">{value}</p>
            <p className="text-sm text-foreground-secondary">{label}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
