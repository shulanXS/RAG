import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { Message } from '@shared/types/chat'

export const useChatStore = defineStore('chat', () => {
  const messages = ref<Message[]>([])
  const isLoading = ref(false)

  function addMessage(message: Message) {
    messages.value.push(message)
  }

  function clearMessages() {
    messages.value = []
  }

  function setLoading(loading: boolean) {
    isLoading.value = loading
  }

  return {
    messages,
    isLoading,
    addMessage,
    clearMessages,
    setLoading,
  }
})
