import { User, Bot } from 'lucide-react'
import { clsx } from 'clsx'

interface MessageItem {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  sources?: Array<{
    doc_id: string
    chunk_id?: string
    title?: string
    content?: string
    score?: number
  }>
}

interface ChatMessageProps {
  message: MessageItem
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'

  return (
    <div className={clsx('flex gap-3', isUser && 'flex-row-reverse')}>
      <div
        className={clsx(
          'w-8 h-8 rounded-full flex items-center justify-center shrink-0 text-xs font-semibold',
          isUser ? 'bg-primary text-white' : 'bg-emerald-500 text-white'
        )}
      >
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      <div className={clsx('flex flex-col gap-1 max-w-[75%]', isUser && 'items-end')}>
        <div
          className={clsx(
            'rounded-lg px-4 py-3 text-sm leading-relaxed',
            isUser
              ? 'bg-primary text-white rounded-tr-none'
              : 'bg-white border border-border rounded-tl-none'
          )}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>

        {message.sources && message.sources.length > 0 && (
          <div className="bg-white border border-border rounded-lg px-3 py-2 text-xs">
            <p className="font-semibold text-foreground-secondary mb-1">Sources:</p>
            {message.sources.map((source, i) => (
              <p key={i} className="text-primary">
                {source.title || source.doc_id}
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
