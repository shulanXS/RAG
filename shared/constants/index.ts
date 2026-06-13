export const API_BASE_URL = '/api'

export const ENDPOINTS = {
  CHAT: '/chat',
  CHAT_STREAM: '/chat/stream',
  SEARCH: '/search',
  DOCUMENTS: '/documents',
  DOCUMENTS_UPLOAD: '/documents/upload',
  HEALTH: '/health',
} as const

export const CHAT_ROLES = {
  USER: 'user',
  ASSISTANT: 'assistant',
  SYSTEM: 'system',
} as const

export const DOCUMENT_STATUS = {
  PENDING: 'pending',
  PROCESSING: 'processing',
  READY: 'ready',
  FAILED: 'failed',
} as const

export const DEFAULT_TOP_K = 5
export const MAX_FILE_SIZE = 50 * 1024 * 1024 // 50MB
