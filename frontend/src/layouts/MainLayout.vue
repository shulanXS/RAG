<template>
  <div class="layout">
    <aside class="sidebar">
      <div class="logo">
        <h1>RAG System</h1>
      </div>
      <nav class="nav">
        <router-link to="/chat" class="nav-item">
          <span>Chat</span>
        </router-link>
        <router-link to="/search" class="nav-item">
          <span>Search</span>
        </router-link>
        <router-link to="/documents" class="nav-item">
          <span>Documents</span>
        </router-link>
        <router-link to="/dashboard" class="nav-item">
          <span>Dashboard</span>
        </router-link>
      </nav>
    </aside>
    <main class="main-content">
      <slot />
    </main>
  </div>
</template>

<style scoped>
.layout {
  display: flex;
  min-height: 100vh;
}

.sidebar {
  width: 240px;
  background: var(--color-bg);
  border-right: 1px solid var(--color-border);
  padding: 16px;
}

.logo h1 {
  font-size: 18px;
  font-weight: 700;
  color: var(--color-primary);
  margin-bottom: 24px;
}

.nav {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-item {
  display: block;
  padding: 10px 12px;
  border-radius: var(--radius);
  color: var(--color-text);
  text-decoration: none;
  transition: background-color 0.15s;
}

.nav-item:hover {
  background: var(--color-bg-secondary);
  text-decoration: none;
}

.nav-item.router-link-active {
  background: var(--color-primary);
  color: white;
}

.main-content {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
}
</style>
