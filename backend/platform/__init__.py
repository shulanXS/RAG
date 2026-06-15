"""
backend.platform — 横切平台层(Sprint 1 重构新增)

与 backend.domain 的关系:
- platform/* 绝不 import domain/*(无环)
- domain/* 接受 platform/* 注入的 port(Tracer / MetricsCollector / CacheStore / Breaker)

迁移状态: Phase 1 建空壳,Phase 2+ 逐步从 backend/middleware、backend/security、backend/observability 迁入
"""
