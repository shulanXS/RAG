import axios from 'axios'
import type {
  ChatRequest,
  ChatResponse,
  SearchRequest,
  SearchResponse,
  DocumentUploadResponse,
} from '@shared/types/api'

const apiClient = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const chatApi = {
  send: async (req: ChatRequest): Promise<ChatResponse> => {
    const { data } = await apiClient.post<ChatResponse>('/chat', req)
    return data
  },

  stream: (req: ChatRequest) => {
    return new EventSource(
      `/api/chat/stream?query=${encodeURIComponent(req.query)}`
    )
  },
}

export const searchApi = {
  query: async (req: SearchRequest): Promise<SearchResponse> => {
    const { data } = await apiClient.post<SearchResponse>('/search', req)
    return data
  },
}

export const documentApi = {
  upload: async (file: File): Promise<DocumentUploadResponse> => {
    const formData = new FormData()
    formData.append('file', file)
    const { data } = await apiClient.post<DocumentUploadResponse>(
      '/documents/upload',
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
      }
    )
    return data
  },

  list: async () => {
    const { data } = await apiClient.get('/documents')
    return data
  },

  delete: async (id: string) => {
    await apiClient.delete(`/documents/${id}`)
  },
}

export default apiClient
