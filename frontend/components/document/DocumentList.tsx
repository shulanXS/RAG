'use client'

import { useEffect, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { documentApi, type DocumentInfo } from '@/lib/api'
import { Trash2 } from 'lucide-react'
import { clsx } from 'clsx'

function formatDate(ts: number) {
  return new Date(ts).toLocaleDateString()
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function DocumentList() {
  const queryClient = useQueryClient()

  const { data, isLoading, error } = useQuery({
    queryKey: ['documents'],
    queryFn: documentApi.list,
  })

  const deleteMutation = useMutation({
    mutationFn: documentApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
  })

  return (
    <div>
      {isLoading && <p className="text-foreground-secondary text-sm">Loading...</p>}
      {error && <p className="text-red-500 text-sm">Failed to load documents</p>}

      {!isLoading && data?.documents.length === 0 && (
        <p className="text-foreground-secondary text-sm text-center py-8">
          No documents uploaded yet.
        </p>
      )}

      {data?.documents && data.documents.length > 0 && (
        <div className="flex flex-col gap-2">
          {data.documents.map((doc: DocumentInfo) => (
            <div
              key={doc.id}
              className="flex items-center justify-between bg-white border border-border rounded-lg px-4 py-3"
            >
              <div>
                <p className="font-medium">{doc.filename}</p>
                <p className="text-xs text-foreground-secondary">
                  {formatDate(doc.uploaded_at)} &middot; {formatSize(doc.size)}
                </p>
              </div>
              <button
                onClick={() => deleteMutation.mutate(doc.id)}
                disabled={deleteMutation.isPending}
                className={clsx(
                  'p-2 rounded-lg text-red-500 hover:bg-red-50 transition-colors',
                  deleteMutation.isPending && 'opacity-50 cursor-not-allowed'
                )}
              >
                <Trash2 size={16} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
