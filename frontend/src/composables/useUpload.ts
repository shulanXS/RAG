import { ref } from 'vue'
import { documentApi } from '@/api'

export function useUpload() {
  const uploading = ref(false)
  const progress = ref(0)
  const error = ref<string | null>(null)

  async function uploadFile(file: File) {
    uploading.value = true
    progress.value = 0
    error.value = null

    try {
      await documentApi.upload(file)
      progress.value = 100
    } catch (e) {
      error.value = e instanceof Error ? e.message : '上传失败'
    } finally {
      uploading.value = false
    }
  }

  return {
    uploading,
    progress,
    error,
    uploadFile,
  }
}
