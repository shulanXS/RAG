// k6/load_test.js — Enterprise RAG 负载测试
// ============================================================================
// 用法:
//   k6 run -e API_URL=https://rag.example.com -e TOKEN=xxx k6/load_test.js
//
// 技术决策:
// - 阶梯式加载: 0→10→50 RPS 阶段，符合生产流量爬升模式
// - SLO 阈值: p95 < 500ms, p99 < 2000ms, 错误率 < 1%
// - 真实场景: 真实 query 列表（中文 + 英文混合），覆盖不同复杂度路径
// ============================================================================

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// 自定义指标
const chatLatency = new Trend('chat_latency_ms');
const errorRate = new Rate('chat_error_rate');
const cacheHitRate = new Trend('chat_cache_hit_rate');
const requestCount = new Counter('chat_requests_total');

export const options = {
  stages: [
    { duration: '30s', target: 10 },   // 预热
    { duration: '1m', target: 50 },    // 正常负载
    { duration: '30s', target: 100 },  // 压力峰值
    { duration: '1m', target: 50 },    // 回归正常
    { duration: '30s', target: 0 },    // 退出
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<2000'],
    http_req_failed: ['rate<0.01'],
    chat_latency_ms: ['p(95)<1500'],
    chat_error_rate: ['rate<0.05'],
  },
  noConnectionReuse: false,
  userAgent: 'k6-rag-loadtest/1.0',
};

const QUERIES = [
  '什么是 RAG 系统？',
  'Explain hybrid search and RRF fusion',
  'How does BM25 differ from dense retrieval?',
  '企业级 RAG 系统的核心组件有哪些？',
  'What is a circuit breaker pattern?',
  'P95 延迟优化的常见手段',
  '请总结 Qdrant 的 HNSW 索引原理',
  'Vector database vs traditional search engine',
  '如何评估一个 RAG 系统的质量？',
  'Give me an example of structured output in LLM',
];

function pickQuery() {
  return QUERIES[Math.floor(Math.random() * QUERIES.length)];
}

export default function () {
  const url = `${__ENV.API_URL}/api/chat`;
  const payload = JSON.stringify({
    query: pickQuery(),
    session_id: `k6-${__VU}-${__ITER}`,
  });
  const params = {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${__ENV.TOKEN}`,
    },
  };

  const res = http.post(url, payload, params);
  requestCount.add(1);
  chatLatency.add(res.timings.duration);
  errorRate.add(res.status !== 200);

  const ok = check(res, {
    'status is 200': (r) => r.status === 200,
    'response has answer': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.answer && body.answer.length > 0;
      } catch (_e) {
        return false;
      }
    },
    'response has sources': (r) => {
      try {
        const body = JSON.parse(r.body);
        return Array.isArray(body.sources) && body.sources.length > 0;
      } catch (_e) {
        return false;
      }
    },
    'response time < 2s': (r) => r.timings.duration < 2000,
  });

  // 模拟用户思考时间
  sleep(1 + Math.random() * 2);
}

// ============================================================================
// 额外场景：流式聊天测试
// ============================================================================

export function streamingChat() {
  const url = `${__ENV.API_URL}/api/stream`;
  const payload = JSON.stringify({
    query: '流式响应测试',
    session_id: `k6-stream-${__VU}`,
  });
  const params = {
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${__ENV.TOKEN}`,
    },
  };

  const res = http.post(url, payload, params);
  check(res, {
    'streaming status is 200': (r) => r.status === 200,
    'content type is SSE': (r) => (r.headers['Content-Type'] || '').includes('event-stream'),
  });

  return res;
}
