<template>
  <MainLayout>
    <div class="search-page">
      <div class="search-header">
        <h2>Search</h2>
      </div>

      <div class="search-form">
        <input
          v-model="query"
          type="text"
          class="search-input"
          placeholder="Enter your search query..."
          @keyup.enter="handleSearch"
        />
        <button class="btn-search" @click="handleSearch" :disabled="loading">
          Search
        </button>
      </div>

      <div v-if="loading" class="loading">Searching...</div>

      <div v-if="results.length > 0" class="results">
        <div
          v-for="result in results"
          :key="result.id"
          class="result-item"
        >
          <div class="result-title">{{ result.title }}</div>
          <div class="result-snippet">{{ result.snippet }}</div>
          <div class="result-score">Score: {{ result.score.toFixed(2) }}</div>
        </div>
      </div>
    </div>
  </MainLayout>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import MainLayout from '@/layouts/MainLayout.vue'
import { searchApi } from '@/api'

const query = ref('')
const loading = ref(false)
const results = ref<any[]>([])

async function handleSearch() {
  if (!query.value.trim()) return

  loading.value = true
  try {
    const response = await searchApi.query({ query: query.value })
    results.value = response.results
  } catch (e) {
    console.error('Search failed:', e)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.search-page {
  max-width: 900px;
  margin: 0 auto;
}

.search-header {
  margin-bottom: 24px;
}

.search-header h2 {
  font-size: 20px;
  font-weight: 600;
}

.search-form {
  display: flex;
  gap: 8px;
  margin-bottom: 24px;
}

.search-input {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  font-size: 14px;
  outline: none;
}

.search-input:focus {
  border-color: var(--color-primary);
}

.btn-search {
  padding: 10px 20px;
  background: var(--color-primary);
  color: white;
  border-radius: var(--radius);
  font-size: 14px;
  font-weight: 500;
}

.btn-search:hover:not(:disabled) {
  background: var(--color-primary-hover);
}

.btn-search:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.loading {
  text-align: center;
  color: var(--color-text-secondary);
}

.results {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.result-item {
  background: var(--color-bg);
  padding: 16px;
  border-radius: var(--radius);
  border: 1px solid var(--color-border);
}

.result-title {
  font-weight: 600;
  margin-bottom: 6px;
}

.result-snippet {
  font-size: 14px;
  color: var(--color-text-secondary);
  margin-bottom: 8px;
}

.result-score {
  font-size: 12px;
  color: var(--color-text-secondary);
}
</style>
