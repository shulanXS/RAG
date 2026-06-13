export interface DocumentInfo {
  id: string
  filename: string
  size: number
  uploadedAt: number
  status: 'pending' | 'processing' | 'ready' | 'failed'
  metadata?: Record<string, unknown>
}

export interface DocumentUploadResponse {
  success: boolean
  document?: DocumentInfo
  error?: string
}

export interface DocumentChunk {
  id: string
  documentId: string
  content: string
  index: number
  metadata?: Record<string, unknown>
}
