import { useState, useRef } from 'react'
import { documentApi } from '@/lib/api'
import { Upload } from 'lucide-react'

export function useUpload() {
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  async function uploadFile(file: File) {
    setUploading(true)
    setProgress(0)
    setError(null)

    try {
      await documentApi.upload(file)
      setProgress(100)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  return { uploading, progress, error, uploadFile }
}
