<template>
  <div class="document-list">
    <div v-if="loading" class="loading">Loading...</div>
    <div v-else-if="documents.length === 0" class="empty">
      No documents uploaded yet.
    </div>
    <div v-else class="list">
      <div
        v-for="doc in documents"
        :key="doc.id"
        class="document-item"
      >
        <div class="doc-info">
          <div class="doc-title">{{ doc.filename }}</div>
          <div class="doc-meta">
            {{ formatDate(doc.uploadedAt) }} &middot; {{ formatSize(doc.size) }}
          </div>
        </div>
        <button class="btn-delete" @click="handleDelete(doc.id)">
          Delete
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { documentApi } from '@/api'
import type { DocumentInfo } from '@shared/types/document'

const documents = ref<DocumentInfo[]>([])
const loading = ref(false)

async function refresh() {
  loading.value = true
  try {
    const data = await documentApi.list()
    documents.value = data.documents || []
  } catch (e) {
    console.error('Failed to load documents:', e)
  } finally {
    loading.value = false
  }
}

async function handleDelete(id: string) {
  try {
    await documentApi.delete(id)
    await refresh()
  } catch (e) {
    console.error('Delete failed:', e)
  }
}

function formatDate(ts: number) {
  return new Date(ts).toLocaleDateString()
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

defineExpose({ refresh })

onMounted(() => {
  refresh()
})
</script>

<style scoped>
.loading,
.empty {
  text-align: center;
  color: var(--color-text-secondary);
  padding: 32px;
}

.list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.document-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  padding: 12px 16px;
}

.doc-title {
  font-weight: 500;
  margin-bottom: 2px;
}

.doc-meta {
  font-size: 12px;
  color: var(--color-text-secondary);
}

.btn-delete {
  padding: 6px 12px;
  color: #ef4444;
  border: 1px solid #ef4444;
  border-radius: var(--radius);
  font-size: 13px;
}

.btn-delete:hover {
  background: #fef2f2;
}
</style>
