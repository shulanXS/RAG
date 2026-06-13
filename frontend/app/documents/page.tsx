'use client'

import { useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { documentApi } from '@/lib/api'
import { Upload } from 'lucide-react'
import { FileUpload } from '@/components/document/FileUpload'
import { DocumentList } from '@/components/document/DocumentList'
import { Button } from '@/components/ui/Button'

export default function DocumentsPage() {
  const queryClient = useQueryClient()
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleFileUpload(file: File) {
    setUploading(true)
    setError(null)
    try {
      await documentApi.upload(file)
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-xl font-semibold mb-6">Documents</h2>

      <div className="mb-6">
        <input
          id="file-upload-input"
          type="file"
          className="hidden"
          onChange={async (e) => {
            const file = e.target.files?.[0]
            if (file) await handleFileUpload(file)
            e.target.value = ''
          }}
        />
        <Button
          onClick={() => document.getElementById('file-upload-input')?.click()}
          disabled={uploading}
        >
          <Upload size={18} />
          <span className="ml-2">
            {uploading ? 'Uploading...' : 'Upload Document'}
          </span>
        </Button>
        {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
      </div>

      <DocumentList />
    </div>
  )
}
