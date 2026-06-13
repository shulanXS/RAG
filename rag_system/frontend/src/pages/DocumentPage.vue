<template>
  <MainLayout>
    <div class="document-page">
      <div class="document-header">
        <h2>Documents</h2>
      </div>

      <FileUpload @uploaded="refresh" />

      <DocumentList ref="listRef" />
    </div>
  </MainLayout>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import MainLayout from '@/layouts/MainLayout.vue'
import FileUpload from '@/components/Document/FileUpload.vue'
import DocumentList from '@/components/Document/DocumentList.vue'

const listRef = ref<InstanceType<typeof DocumentList> | null>(null)

function refresh() {
  listRef.value?.refresh()
}
</script>

<style scoped>
.document-page {
  max-width: 900px;
  margin: 0 auto;
}

.document-header {
  margin-bottom: 24px;
}

.document-header h2 {
  font-size: 20px;
  font-weight: 600;
}
</style>
