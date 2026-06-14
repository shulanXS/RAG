'use client'

import { Bot } from 'lucide-react'
import { clsx } from 'clsx'
import { useState, useEffect } from 'react'

interface Citation {
  doc_id: string
  chunk_id?: string
  content?: string
  score?: number
  is_grounded?: boolean
  supported_claims?: string[]
  reason?: string
}

interface StreamingMessageProps {
  content: string
  sources?: Citation[]
  isStreaming?: boolean
  onFinish?: () => void
}

export function StreamingMessage({ content, sources, isStreaming = false }: StreamingMessageProps) {
  const [expandedSources, setExpandedSources] = useState(false)

  return (
    <div className="flex gap-3">
      <div className="w-8 h-8 rounded-full bg-emerald-500 text-white flex items-center justify-center shrink-0">
        <Bot size={16} />
      </div>
      <div className="flex flex-col gap-1 max-w-[75%]">
        <div className="bg-white border border-border rounded-lg px-4 py-3 text-sm leading-relaxed rounded-tl-none">
          <p className="whitespace-pre-wrap">{content}</p>
          {isStreaming && (
            <span className="inline-block w-2 h-4 bg-primary ml-1 animate-pulse" />
          )}
        </div>

        {sources && sources.length > 0 && (
          <div className="mt-1">
            <button
              onClick={() => setExpandedSources(!expandedSources)}
              className="text-xs text-primary hover:underline"
            >
              {expandedSources ? 'Hide' : 'Show'} references ({sources.length})
            </button>

            {expandedSources && (
              <div className="mt-2 space-y-2">
                {sources.map((source, i) => (
                  <div
                    key={i}
                    className={clsx(
                      'bg-white border rounded-lg px-3 py-2 text-xs',
                      source.is_grounded === false && 'border-yellow-300 bg-yellow-50'
                    )}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-foreground-secondary">
                        {source.chunk_id || source.doc_id}
                      </span>
                      {source.score !== undefined && (
                        <span className="text-foreground-secondary">
                          {source.score.toFixed(3)}
                        </span>
                      )}
                    </div>
                    <p className="text-foreground-secondary line-clamp-3">
                      {source.content || source.quote}
                    </p>
                    {source.is_grounded === false && (
                      <p className="mt-1 text-yellow-700 text-xs">
                        {source.reason || 'Unverified citation'}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
