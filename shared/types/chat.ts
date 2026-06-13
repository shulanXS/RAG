export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  sources?: Source[]
}

export interface Source {
  id: string
  title: string
  content: string
  score?: number
  metadata?: Record<string, unknown>
}
