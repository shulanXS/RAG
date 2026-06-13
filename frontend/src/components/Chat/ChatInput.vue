<template>
  <div class="chat-input">
    <textarea
      v-model="input"
      class="input-field"
      placeholder="Type your message..."
      :disabled="disabled"
      @keydown.enter.exact.prevent="handleSubmit"
      rows="1"
    />
    <button
      class="btn-send"
      @click="handleSubmit"
      :disabled="disabled || !input.trim()"
    >
      Send
    </button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  disabled?: boolean
}>()

const emit = defineEmits<{
  submit: [content: string]
}>()

const input = ref('')

function handleSubmit() {
  if (!input.value.trim()) return
  emit('submit', input.value)
  input.value = ''
}
</script>

<style scoped>
.chat-input {
  display: flex;
  gap: 8px;
  padding: 16px;
  background: var(--color-bg);
  border-top: 1px solid var(--color-border);
  border-radius: var(--radius);
}

.input-field {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  font-size: 14px;
  resize: none;
  outline: none;
  font-family: inherit;
  line-height: 1.5;
}

.input-field:focus {
  border-color: var(--color-primary);
}

.input-field:disabled {
  background: var(--color-bg-secondary);
  cursor: not-allowed;
}

.btn-send {
  padding: 10px 20px;
  background: var(--color-primary);
  color: white;
  border-radius: var(--radius);
  font-size: 14px;
  font-weight: 500;
  align-self: flex-end;
}

.btn-send:hover:not(:disabled) {
  background: var(--color-primary-hover);
}

.btn-send:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
