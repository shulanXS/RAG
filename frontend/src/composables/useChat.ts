import { ref } from 'vue'
import { useChatStore } from '@/stores/chat'
import { chatApi } from '@/api'
import type { Message } from '@shared/types/chat'

export function useChat() {
  const chatStore = useChatStore()

  async function sendMessage(content: string) {
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: Date.now(),
    }
    chatStore.addMessage(userMessage)
    chatStore.setLoading(true)

    try {
      const response = await chatApi.send({ query: content })

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.answer,
        timestamp: Date.now(),
        sources: response.sources,
      }
      chatStore.addMessage(assistantMessage)
    } catch (error) {
      const errorMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '抱歉，发生了错误。请稍后重试。',
        timestamp: Date.now(),
      }
      chatStore.addMessage(errorMessage)
    } finally {
      chatStore.setLoading(false)
    }
  }

  function clearChat() {
    chatStore.clearMessages()
  }

  return {
    messages: chatStore.messages,
    isLoading: chatStore.isLoading,
    sendMessage,
    clearChat,
  }
}
