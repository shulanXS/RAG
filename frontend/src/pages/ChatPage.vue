<template>
  <MainLayout>
    <div class="chat-page">
      <div class="chat-header">
        <h2>Chat</h2>
        <button class="btn-clear" @click="clearChat">Clear</button>
      </div>

      <div class="messages">
        <ChatMessage
          v-for="msg in messages"
          :key="msg.id"
          :message="msg"
        />
      </div>

      <ChatInput
        :disabled="isLoading"
        @submit="handleSubmit"
      />
    </div>
  </MainLayout>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import MainLayout from '@/layouts/MainLayout.vue'
import ChatMessage from '@/components/Chat/ChatMessage.vue'
import ChatInput from '@/components/Chat/ChatInput.vue'
import { useChat } from '@/composables/useChat'

const { messages, isLoading, sendMessage, clearChat } = useChat()

async function handleSubmit(content: string) {
  await sendMessage(content)
}
</script>

<style scoped>
.chat-page {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 48px);
  max-width: 900px;
  margin: 0 auto;
}

.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.chat-header h2 {
  font-size: 20px;
  font-weight: 600;
}

.btn-clear {
  padding: 6px 12px;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  font-size: 13px;
  color: var(--color-text-secondary);
}

.btn-clear:hover {
  background: var(--color-bg-secondary);
}

.messages {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding-bottom: 16px;
}
</style>
