'use client'

import { useState } from 'react'
import { useSearch } from '@/hooks/useSearch'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Search } from 'lucide-react'

export default function SearchPage() {
  const { query, results, loading, handleSearch } = useSearch()
  const [inputValue, setInputValue] = useState('')

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (inputValue.trim()) {
      handleSearch(inputValue.trim())
    }
  }

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-xl font-semibold mb-6">Search</h2>

      <form onSubmit={handleSubmit} className="flex gap-2 mb-6">
        <Input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Enter your search query..."
          className="flex-1"
        />
        <Button type="submit" disabled={loading}>
          <Search size={18} />
        </Button>
      </form>

      {loading && (
        <p className="text-foreground-secondary text-sm">Searching...</p>
      )}

      {results.length > 0 && (
        <div className="flex flex-col gap-3">
          {results.map((result) => (
            <div
              key={result.id}
              className="bg-white border border-border rounded-lg p-4"
            >
              <p className="font-semibold mb-1">{result.title || result.id}</p>
              <p className="text-sm text-foreground-secondary mb-2">
                {result.snippet}
              </p>
              {result.score !== undefined && (
                <p className="text-xs text-foreground-secondary">
                  Score: {result.score.toFixed(2)}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
