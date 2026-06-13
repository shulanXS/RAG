<template>
  <div class="chat-message" :class="[`role-${message.role}`]">
    <div class="avatar">
      {{ message.role === 'user' ? 'U' : 'AI' }}
    </div>
    <div class="content">
      <div class="text">{{ message.content }}</div>
      <div v-if="message.sources && message.sources.length > 0" class="sources">
        <div class="sources-label">Sources:</div>
        <div
          v-for="source in message.sources"
          :key="source.id"
          class="source-item"
        >
          {{ source.title }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { Message } from '@shared/types/chat'

defineProps<{
  message: Message
}>()
</script>

<style scoped>
.chat-message {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}

.avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 600;
  flex-shrink: 0;
}

.role-user .avatar {
  background: var(--color-primary);
  color: white;
}

.role-assistant .avatar {
  background: #10b981;
  color: white;
}

.content {
  flex: 1;
  min-width: 0;
}

.text {
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  padding: 12px 16px;
  font-size: 14px;
  line-height: 1.6;
}

.role-user .text {
  background: var(--color-primary);
  border-color: var(--color-primary);
  color: white;
}

.sources {
  margin-top: 8px;
  padding: 8px 12px;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  font-size: 12px;
}

.sources-label {
  font-weight: 600;
  margin-bottom: 4px;
  color: var(--color-text-secondary);
}

.source-item {
  padding: 2px 0;
  color: var(--color-primary);
}
</style>
