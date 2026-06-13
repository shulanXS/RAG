import { useState } from 'react'
import { searchApi, type SearchResult } from '@/lib/api'

export function useSearch() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function search(q: string) {
    if (!q.trim()) return

    setLoading(true)
    setError(null)
    try {
      const res = await searchApi.query(q)
      setResults(res.results)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  function handleSearch(q: string) {
    setQuery(q)
    search(q)
  }

  return { query, results, loading, error, handleSearch }
}
