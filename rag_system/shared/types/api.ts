import type { Message } from './chat'
import type { DocumentInfo } from './document'

export interface ChatRequest {
  query: string
  sessionId?: string
  history?: Message[]
  metadata?: Record<string, unknown>
}

export interface ChatResponse {
  answer: string
  sources: ChatResponseSource[]
  sessionId?: string
  metadata?: Record<string, unknown>
}

export interface ChatResponseSource {
  id: string
  title: string
  content: string
  score: number
  metadata?: Record<string, unknown>
}

export interface SearchRequest {
  query: string
  topK?: number
  filters?: Record<string, unknown>
}

export interface SearchResponse {
  results: SearchResult[]
  total: number
  metadata?: Record<string, unknown>
}

export interface SearchResult {
  id: string
  title: string
  snippet: string
  score: number
  metadata?: Record<string, unknown>
}

export interface ApiError {
  code: string
  message: string
  details?: Record<string, unknown>
}

export interface DocumentListResponse {
  documents: DocumentInfo[]
  total: number
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy'
  version: string
  uptime: number
  components?: Record<string, { status: string; latency?: number }>
}
