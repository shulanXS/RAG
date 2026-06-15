"""
backend.domain — 领域层(Sprint 1 重构新增)

设计原则(见 plan §1.1):
- 只关心"做什么",不关心"如何观测/限流/认证"
- 跨切面(circuit breaker、context、tracing)由 platform/* 注入
- 内部可互相依赖但只能向下(retrieval 不能 import agent)

迁移状态:
- Phase 1 (Sprint 1): 建空壳,作为未来代码落点
- Phase 2 (后续 Sprint): 逐步从 backend/agentic、backend/retrieval、backend/cache 迁入
- Phase 3: 旧包改为 shim re-export
- Phase 4: 删旧包
"""
