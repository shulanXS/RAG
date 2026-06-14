"use client";

/**
 * Traces (P1-B22)
 * - 2026-06-14: 重构为 Jaeger UI 跳转
 * - 之前的自定义 trace viewer (基于内存 ring buffer) 已删除
 * - 真实 span 时长由 OTel 上报到 Jaeger，统一在 Jaeger UI 查询
 */
import { useEffect, useState } from 'react';

const JAEGER_URL = 'http://localhost:16686';

export default function TracesPage() {
  const [token, setToken] = useState<string>('');
  const [jaegerReady, setJaegerReady] = useState<boolean | null>(null);

  useEffect(() => {
    setToken(localStorage.getItem('rag_token') || '');
    // 健康检查：Jaeger UI 是否可达
    fetch(`${JAEGER_URL}/api/services`)
      .then((r) => setJaegerReady(r.ok))
      .catch(() => setJaegerReady(false));
  }, []);

  return (
    <div
      style={{
        padding: 32,
        fontFamily: 'system-ui, sans-serif',
        maxWidth: 720,
        margin: '0 auto',
      }}
    >
      <h2 style={{ margin: 0 }}>Traces</h2>
      <p style={{ color: '#6b7280', fontSize: 14, marginTop: 8 }}>
        P1-B22: 之前的自定义 trace viewer 已删除。所有 span 通过 OpenTelemetry
        上报到 Jaeger，统一在 Jaeger UI 查询。
      </p>

      <div
        style={{
          marginTop: 24,
          padding: 16,
          background: '#f9fafb',
          borderRadius: 8,
          border: '1px solid #e5e7eb',
        }}
      >
        <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 8 }}>
          Jaeger UI 状态
        </div>
        {jaegerReady === null && (
          <div style={{ color: '#9ca3af' }}>检测中...</div>
        )}
        {jaegerReady === true && (
          <div style={{ color: '#10b981' }}>✓ Jaeger 已就绪 (port 16686)</div>
        )}
        {jaegerReady === false && (
          <div style={{ color: '#ef4444' }}>
            ✗ Jaeger 不可达。请确认 docker-compose 中 jaeger 服务已启动。
          </div>
        )}
        <div style={{ marginTop: 16, display: 'flex', gap: 8 }}>
          <a
            href={`${JAEGER_URL}/search?service=backend`}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              padding: '8px 16px',
              background: '#1f2937',
              color: '#fff',
              borderRadius: 4,
              textDecoration: 'none',
              fontSize: 14,
            }}
          >
            打开 Jaeger UI →
          </a>
          <a
            href={`${JAEGER_URL}/search?service=backend&tags=%7B%22component%22%3A%22rag%22%7D`}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              padding: '8px 16px',
              background: '#f3f4f6',
              color: '#000',
              borderRadius: 4,
              textDecoration: 'none',
              fontSize: 14,
            }}
          >
            只看 RAG span
          </a>
        </div>
      </div>

      <div
        style={{
          marginTop: 24,
          padding: 16,
          background: '#eff6ff',
          borderRadius: 8,
          fontSize: 13,
          color: '#1e40af',
        }}
      >
        <strong>迁移说明：</strong>
        <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20 }}>
          <li>旧路径 <code>/api/traces</code> 已删除 (P1-B22)</li>
          <li>真实 span 时长由 OTel 记录，duration_ms 不会全是 0</li>
          <li>production: 用 OTLP exporter 把 trace 推送到 Jaeger / Tempo / Honeycomb</li>
          <li>本地 dev: <code>make up</code> 启动 docker-compose 后 Jaeger 自动运行</li>
        </ul>
      </div>
    </div>
  );
}
