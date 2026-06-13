<template>
  <MainLayout>
    <div class="dashboard-page">
      <h2>Dashboard</h2>
      <div class="stats">
        <div class="stat-card">
          <div class="stat-value">{{ stats.documents }}</div>
          <div class="stat-label">Documents</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ stats.queries }}</div>
          <div class="stat-label">Queries Today</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ stats.cacheHitRate }}%</div>
          <div class="stat-label">Cache Hit Rate</div>
        </div>
      </div>
    </div>
  </MainLayout>
</template>

<script setup lang="ts">
import { reactive } from 'vue'
import MainLayout from '@/layouts/MainLayout.vue'

const stats = reactive({
  documents: 0,
  queries: 0,
  cacheHitRate: 0,
})
</script>

<style scoped>
.dashboard-page h2 {
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 24px;
}

.stats {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
}

.stat-card {
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  padding: 20px;
}

.stat-value {
  font-size: 32px;
  font-weight: 700;
  color: var(--color-primary);
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: var(--color-text-secondary);
}
</style>
