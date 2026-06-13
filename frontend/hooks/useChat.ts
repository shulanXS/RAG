import { useChatStore } from '@/store/chatStore'
import { chatApi, type Message } from '@/lib/api'

export function useChat() {
  const { messages, isLoading, addMessage, clearMessages, setLoading } = useChatStore()

  async function sendMessage(content: string) {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: Date.now(),
    }
    addMessage(userMsg)
    setLoading(true)

    try {
      const res = await chatApi.send({ query: content })
      addMessage({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: res.answer,
        timestamp: Date.now(),
        sources: res.sources,
      })
    } catch {
      addMessage({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '抱歉，发生了错误。请稍后重试。',
        timestamp: Date.now(),
      })
    } finally {
      setLoading(false)
    }
  }

  return { messages, isLoading, sendMessage, clearMessages }
}
