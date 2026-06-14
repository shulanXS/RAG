'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { Bot, User, Trash2 } from 'lucide-react'
import { StreamingMessage } from '@/components/chat/StreamingMessage'
import { ChatInput } from '@/components/chat/ChatInput'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  sources?: any[]
  isStreaming?: boolean
}

const SESSION_KEY = 'rag_session_id'
const TOKEN_KEY = 'rag_access_token'

function getSessionId(): string {
  if (typeof window === 'undefined') return ''
  let sid = sessionStorage.getItem(SESSION_KEY)
  if (!sid) {
    sid = crypto.randomUUID()
    sessionStorage.setItem(SESSION_KEY, sid)
  }
  return sid
}

function getToken(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem(TOKEN_KEY)
}

function setToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token)
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId] = useState<string>(getSessionId())
  const [authError, setAuthError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const stopStream = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
    setIsLoading(false)
  }, [])

  async function handleStream(query: string) {
    if (!query.trim()) return

    const token = getToken()
    if (!token) {
      setAuthError('Please login first')
      return
    }

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
      timestamp: Date.now(),
    }
    const assistantMsg: Message = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      isStreaming: true,
    }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setIsLoading(true)
    setAuthError(null)
    abortRef.current = new AbortController()

    try {
      const response = await fetch('/api/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ query, session_id: sessionId }),
        signal: abortRef.current.signal,
      })

      if (!response.ok) {
        if (response.status === 401) {
          setAuthError('Session expired. Please login again.')
          setToken('')
        }
        throw new Error(`HTTP ${response.status}`)
      }

      const reader = response.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6)
          if (raw === '[DONE]' || raw === '[DONE]\n') continue

          try {
            const event = JSON.parse(raw)

            if (event.stage === 'generating' && event.token) {
              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantMsg.id
                    ? { ...m, content: m.content + event.token }
                    : m
                )
              )
            } else if (event.stage === 'done') {
              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantMsg.id
                    ? {
                        ...m,
                        content: event.answer || m.content,
                        sources: event.citations || [],
                        isStreaming: false,
                      }
                    : m
                )
              )
            } else if (event.stage === 'error') {
              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantMsg.id
                    ? { ...m, content: `Error: ${event.message}`, isStreaming: false }
                    : m
                )
              )
            }
          } catch {
            // skip malformed events
          }
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') return
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantMsg.id
            ? { ...m, content: `Request failed: ${err.message}`, isStreaming: false }
            : m
        )
      )
    } finally {
      abortRef.current = null
      setIsLoading(false)
    }
  }

  function handleSubmit(content: string) {
    if (isLoading) return
    handleStream(content)
  }

  function handleClear() {
    stopStream()
    setMessages([])
  }

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto w-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 shrink-0">
        <div className="flex items-center gap-2">
          <Bot size={20} className="text-primary" />
          <h2 className="text-xl font-semibold">Chat</h2>
          <span className="text-xs text-foreground-secondary bg-background-secondary px-2 py-0.5 rounded-full">
            {sessionId.slice(0, 8)}...
          </span>
        </div>
        <button
          onClick={handleClear}
          className="flex items-center gap-1.5 text-sm text-foreground-secondary hover:text-foreground px-3 py-1.5 rounded-lg hover:bg-background-secondary transition-colors"
        >
          <Trash2 size={14} />
          Clear
        </button>
      </div>

      {/* Auth error */}
      {authError && (
        <div className="mb-4 px-4 py-3 bg-yellow-50 border border-yellow-300 rounded-lg text-sm text-yellow-800">
          {authError}
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto flex flex-col gap-4 mb-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-foreground-secondary">
            <Bot size={40} className="mb-3 opacity-30" />
            <p className="text-sm">Start a conversation by typing a message below.</p>
          </div>
        )}

        {messages.map(msg =>
          msg.role === 'user' ? (
            <div key={msg.id} className="flex gap-3 flex-row-reverse">
              <div className="w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center shrink-0">
                <User size={16} />
              </div>
              <div className="bg-primary text-white rounded-lg px-4 py-3 text-sm max-w-[75%] rounded-tr-none">
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          ) : (
            <StreamingMessage
              key={msg.id}
              content={msg.content}
              sources={msg.sources}
              isStreaming={msg.isStreaming}
            />
          )
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="shrink-0">
        <ChatInput disabled={isLoading} onSubmit={handleSubmit} />
      </div>
    </div>
  )
}
