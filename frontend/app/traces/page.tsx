"use client";

/**
 * Trace Viewer (P2.2)
 * - 列出最近 100 条 trace
 * - 详情页显示 waterfall (cache_check → query_rewrite → routing → retrieval → generation → reflection)
 * - 按 complexity 过滤
 */
import { useEffect, useState } from 'react';

interface TraceSummary {
  trace_id: string;
  started_at_ms: number;
  ended_at_ms: number;
  latency_ms: number;
  complexity: string;
  routing_confidence: number;
  cache_hit: boolean;
  answer_length: number;
  span_count: number;
}

interface Span {
  name: string;
  duration_ms: number;
  attrs: Record<string, any>;
}

interface TraceDetail extends TraceSummary {
  spans: Span[];
}

const COMPLEXITY_COLORS: Record<string, string> = {
  simple: '#10b981',
  moderate: '#f59e0b',
  complex: '#ef4444',
  beyond_kb: '#6366f1',
};

export default function TracesPage() {
  const [traces, setTraces] = useState<TraceSummary[]>([]);
  const [selected, setSelected] = useState<TraceDetail | null>(null);
  const [filter, setFilter] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [token, setToken] = useState<string>('');

  useEffect(() => {
    const t = localStorage.getItem('rag_token') || '';
    setToken(t);
  }, []);

  async function loadTraces() {
    if (!token) return;
    setLoading(true);
    try {
      const url = filter ? `/api/traces?complexity=${filter}&limit=100` : '/api/traces?limit=100';
      const res = await fetch(url, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      setTraces(data.traces || []);
    } catch (e) {
      console.error('loadTraces failed', e);
    } finally {
      setLoading(false);
    }
  }

  async function loadDetail(traceId: string) {
    if (!token) return;
    try {
      const res = await fetch(`/api/traces/${traceId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      setSelected(data);
    } catch (e) {
      console.error('loadDetail failed', e);
    }
  }

  useEffect(() => {
    loadTraces();
    const t = setInterval(loadTraces, 5000);
    return () => clearInterval(t);
  }, [token, filter]);

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'system-ui, sans-serif' }}>
      <aside style={{ width: 380, overflowY: 'auto', borderRight: '1px solid #e5e7eb' }}>
        <div style={{ padding: 16, borderBottom: '1px solid #e5e7eb' }}>
          <h2 style={{ margin: 0, fontSize: 18 }}>Traces</h2>
          <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
            {['', 'simple', 'moderate', 'complex'].map((c) => (
              <button
                key={c}
                onClick={() => setFilter(c)}
                style={{
                  padding: '4px 8px',
                  fontSize: 12,
                  background: filter === c ? '#1f2937' : '#f3f4f6',
                  color: filter === c ? '#fff' : '#000',
                  border: 'none',
                  borderRadius: 4,
                  cursor: 'pointer',
                }}
              >
                {c || 'all'}
              </button>
            ))}
          </div>
        </div>
        {loading && <p style={{ padding: 12, color: '#9ca3af' }}>loading...</p>}
        {traces.map((t) => (
          <div
            key={t.trace_id}
            onClick={() => loadDetail(t.trace_id)}
            style={{
              padding: 12,
              borderBottom: '1px solid #f3f4f6',
              cursor: 'pointer',
              background: selected?.trace_id === t.trace_id ? '#eff6ff' : '#fff',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span
                style={{
                  fontSize: 11,
                  padding: '2px 6px',
                  background: COMPLEXITY_COLORS[t.complexity] || '#9ca3af',
                  color: '#fff',
                  borderRadius: 4,
                }}
              >
                {t.complexity}
              </span>
              {t.cache_hit && (
                <span style={{ fontSize: 10, color: '#10b981' }}>cache</span>
              )}
            </div>
            <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
              {t.latency_ms.toFixed(0)}ms · {t.span_count} spans · conf {t.routing_confidence.toFixed(2)}
            </div>
            <div style={{ fontSize: 10, color: '#9ca3af', marginTop: 2 }}>
              {new Date(t.started_at_ms).toLocaleTimeString()} · {t.trace_id.slice(0, 8)}
            </div>
          </div>
        ))}
      </aside>

      <main style={{ flex: 1, overflowY: 'auto', padding: 24 }}>
        {!selected && <p style={{ color: '#9ca3af' }}>Select a trace to view details</p>}
        {selected && (
          <>
            <h2 style={{ margin: 0 }}>Trace {selected.trace_id.slice(0, 12)}...</h2>
            <div style={{ marginTop: 8, color: '#6b7280', fontSize: 14 }}>
              complexity: {selected.complexity} · cache_hit: {String(selected.cache_hit)} · latency:{' '}
              {selected.latency_ms.toFixed(0)}ms · answer length: {selected.answer_length}
            </div>
            <h3 style={{ marginTop: 24 }}>Waterfall</h3>
            <div style={{ marginTop: 12 }}>
              {selected.spans.length === 0 && (
                <p style={{ color: '#9ca3af' }}>no spans</p>
              )}
              {selected.spans.map((s, i) => (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    marginTop: 8,
                    padding: 8,
                    background: '#f9fafb',
                    borderRadius: 4,
                  }}
                >
                  <div style={{ width: 24, color: '#9ca3af', fontSize: 12 }}>{i + 1}</div>
                  <div style={{ fontSize: 13, fontWeight: 500, width: 200 }}>{s.name}</div>
                  <div
                    style={{
                      flex: 1,
                      height: 8,
                      background: '#e5e7eb',
                      borderRadius: 4,
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        width: `${Math.min(100, (s.duration_ms / Math.max(1, selected.latency_ms)) * 100)}%`,
                        height: '100%',
                        background: '#3b82f6',
                      }}
                    />
                  </div>
                  <div style={{ fontSize: 12, color: '#6b7280', width: 60, textAlign: 'right' }}>
                    {s.duration_ms > 0 ? `${s.duration_ms.toFixed(0)}ms` : '—'}
                  </div>
                </div>
              ))}
            </div>
            <h3 style={{ marginTop: 24 }}>Span Attributes</h3>
            <pre
              style={{
                background: '#1f2937',
                color: '#f9fafb',
                padding: 16,
                borderRadius: 4,
                fontSize: 12,
                overflow: 'auto',
              }}
            >
              {JSON.stringify(selected.spans, null, 2)}
            </pre>
          </>
        )}
      </main>
    </div>
  );
}
