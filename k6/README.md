# k6 Load Testing — Enterprise RAG

## 概述

`load_test.js` 是一个针对 RAG 系统的可扩展 k6 负载测试脚本，用于在生产
部署前/后验证系统 SLO 是否达标。

## 用法

### 前置依赖

```bash
# 安装 k6
brew install k6           # macOS
apt install k6            # Debian/Ubuntu
choco install k6          # Windows

# 准备环境变量
export API_URL="http://localhost:8000"
export TOKEN="<your-jwt-token>"
```

### 基础运行

```bash
k6 run k6/load_test.js
```

### 自定义 VU 数与持续时间

```bash
k6 run --vus 100 --duration 5m k6/load_test.js
```

### 输出结果到 InfluxDB / Prometheus

```bash
# InfluxDB
k6 run --out influxdb=http://localhost:8086/k6 k6/load_test.js

# Prometheus
k6 run --out prometheus=... k6/load_test.js
```

## 场景设计

| 阶段 | 时长 | VU 数 | 含义 |
|------|------|------|------|
| 预热 | 30s | 0→10 | 让 JIT/连接池就绪 |
| 正常 | 1m | 10→50 | 模拟生产典型负载 |
| 峰值 | 30s | 50→100 | 模拟流量尖刺 |
| 回归 | 1m | 100→50 | 系统从峰值恢复 |
| 退出 | 30s | 50→0 | 优雅降压 |

## SLO 阈值

| 指标 | 阈值 | 含义 |
|------|------|------|
| `http_req_duration p95` | < 500ms | 95% 请求应在 500ms 内完成 |
| `http_req_duration p99` | < 2000ms | 99% 请求应在 2s 内完成 |
| `http_req_failed` | < 1% | 错误率应低于 1% |
| `chat_latency_ms p95` | < 1500ms | 端到端（含 LLM）P95 |
| `chat_error_rate` | < 5% | 包含 LLM 失败的可接受错误率 |

## 关键指标

- `chat_requests_total`: 总请求数
- `chat_latency_ms`: 聊天请求时延分布
- `chat_error_rate`: 错误率
- `chat_cache_hit_rate`: 语义缓存命中率（手动从响应中计算）

## 解读结果

```bash
# 报告摘要
checks.........................: 100.00% ✓ 0 ✗
data_received..................: 1.2 MB 13 kB/s
data_sent......................: 230 kB 2.5 kB/s
http_req_blocked...............: avg=1.2ms    max=15ms
http_req_duration..............: avg=120ms    p(95)=450ms
http_req_failed................: 0.00%  ✓ 0   ✗ 1500
http_req_receiving.............: avg=0.5ms    max=5ms
http_req_tls_handshaking.......: avg=0ms      max=0ms
http_req_waiting...............: avg=119ms    max=480ms
http_reqs......................: 1500   16.66/s
iteration_duration.............: avg=2.5s     max=5s
iterations.....................: 1500   16.66/s
vus............................: 0      min=0 max=50
```

✅ 所有 thresholds 通过 = 系统满足 SLO  
❌ 任何 threshold 失败 = 需在 SLO 退化前扩容/调优

## 进阶

### 添加新场景

```javascript
// 添加 SSE 流式测试
import { streamingChat } from './load_test.js';

export const options = {
  scenarios: {
    sync_chat: { executor: 'constant-vus', vus: 30, duration: '5m', fn: default },
    stream_chat: { executor: 'constant-vus', vus: 10, duration: '5m', fn: streamingChat },
  },
};
```

### 在 CI 中集成

```yaml
- name: 负载测试
  run: |
    docker run --rm -v $PWD:/scripts grafana/k6:latest run /scripts/k6/load_test.js
```
