<template>
  <div class="file-upload">
    <input
      ref="fileInput"
      type="file"
      class="file-input"
      @change="handleFileChange"
    />
    <button class="btn-upload" @click="triggerFileInput">
      {{ uploading ? `Uploading... ${progress}%` : 'Upload Document' }}
    </button>
    <span v-if="error" class="error">{{ error }}</span>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useUpload } from '@/composables/useUpload'

const emit = defineEmits<{
  uploaded: []
}>()

const fileInput = ref<HTMLInputElement | null>(null)
const { uploading, progress, error, uploadFile } = useUpload()

function triggerFileInput() {
  fileInput.value?.click()
}

async function handleFileChange(e: Event) {
  const target = e.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return

  await uploadFile(file)
  emit('uploaded')

  target.value = ''
}
</script>

<style scoped>
.file-upload {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 24px;
}

.file-input {
  display: none;
}

.btn-upload {
  padding: 8px 16px;
  background: var(--color-primary);
  color: white;
  border-radius: var(--radius);
  font-size: 14px;
}

.btn-upload:hover {
  background: var(--color-primary-hover);
}

.error {
  color: #ef4444;
  font-size: 13px;
}
</style>
