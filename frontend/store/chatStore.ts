import { create } from 'zustand'

export interface Message {
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

interface ChatStore {
  messages: Message[]
  isLoading: boolean
  addMessage: (msg: Message) => void
  clearMessages: () => void
  setLoading: (loading: boolean) => void
}

export const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  isLoading: false,
  addMessage: (msg) =>
    set((state) => ({ messages: [...state.messages, msg] })),
  clearMessages: () => set({ messages: [] }),
  setLoading: (loading) => set({ isLoading: loading }),
}))
