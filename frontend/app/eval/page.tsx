"use client";

/**
 * Evaluation Dashboard (P2.3)
 * - 5 RAGAS 指标 trend + pass rate + p95 latency
 * - "Run Now" 按钮
 */
import { useEffect, useState } from 'react';

interface EvalRun {
  run_id: string;
  started_at: string;
  ended_at: string;
  total_cases: number;
  passed_cases: number;
  avg_faithfulness: number;
  avg_answer_relevancy: number;
  avg_context_precision: number;
  avg_context_recall: number;
  avg_answer_correctness: number;
  weakest_metric: string;
}

export default function EvalPage() {
  const [latest, setLatest] = useState<any>(null);
  const [runs, setRuns] = useState<EvalRun[]>([]);
  const [triggering, setTriggering] = useState(false);
  const [token, setToken] = useState('');

  useEffect(() => {
    setToken(localStorage.getItem('rag_token') || '');
  }, []);

  async function fetchSummary() {
    if (!token) return;
    try {
      const res = await fetch('/api/eval/summary?window=24h', {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      setLatest(data);
    } catch (e) {
      console.error(e);
    }
  }

  async function fetchRuns() {
    if (!token) return;
    try {
      const res = await fetch('/api/eval/runs?limit=50', {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      setRuns(data.runs || []);
    } catch (e) {
      console.error(e);
    }
  }

  async function triggerRun() {
    if (!token) return;
    setTriggering(true);
    try {
      await fetch('/api/eval/run', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      setTimeout(() => {
        fetchSummary();
        fetchRuns();
        setTriggering(false);
      }, 3000);
    } catch (e) {
      console.error(e);
      setTriggering(false);
    }
  }

  useEffect(() => {
    fetchSummary();
    fetchRuns();
  }, [token]);

  if (!latest?.available) {
    return (
      <div style={{ padding: 24, fontFamily: 'system-ui' }}>
        <h1>Evaluation Dashboard</h1>
        <p>No eval runs yet.</p>
        <button
          onClick={triggerRun}
          disabled={triggering}
          style={{
            padding: '8px 16px',
            background: '#1f2937',
            color: '#fff',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer',
          }}
        >
          {triggering ? 'Running...' : 'Run Now'}
        </button>
      </div>
    );
  }

  const passRate = latest.pass_rate ?? 0;
  const metrics = [
    { key: 'avg_faithfulness', label: 'Faithfulness', target: 0.85 },
    { key: 'avg_answer_relevancy', label: 'Answer Relevancy', target: 0.75 },
    { key: 'avg_context_precision', label: 'Context Precision', target: 0.70 },
    { key: 'avg_context_recall', label: 'Context Recall', target: 0.70 },
    { key: 'avg_answer_correctness', label: 'Answer Correctness', target: 0.80 },
  ];

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1>Evaluation Dashboard</h1>
        <button
          onClick={triggerRun}
          disabled={triggering}
          style={{
            padding: '8px 16px',
            background: triggering ? '#9ca3af' : '#1f2937',
            color: '#fff',
            border: 'none',
            borderRadius: 4,
            cursor: triggering ? 'not-allowed' : 'pointer',
          }}
        >
          {triggering ? 'Running...' : 'Run Now'}
        </button>
      </div>

      <div style={{ display: 'flex', gap: 16, marginTop: 16 }}>
        <Stat label="Pass Rate" value={`${(passRate * 100).toFixed(1)}%`} />
        <Stat label="Total Cases" value={latest.total_cases} />
        <Stat label="Passed" value={latest.passed_cases} />
        <Stat label="Weakest" value={latest.weakest_metric} />
      </div>

      <h2 style={{ marginTop: 32 }}>RAGAS Metrics (current)</h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12, marginTop: 12 }}>
        {metrics.map((m) => {
          const v = latest[m.key] ?? 0;
          const ok = v >= m.target;
          return (
            <div
              key={m.key}
              style={{
                padding: 16,
                background: ok ? '#ecfdf5' : '#fef2f2',
                border: `1px solid ${ok ? '#10b981' : '#ef4444'}`,
                borderRadius: 4,
              }}
            >
              <div style={{ fontSize: 12, color: '#6b7280' }}>{m.label}</div>
              <div style={{ fontSize: 24, fontWeight: 600, color: ok ? '#10b981' : '#ef4444' }}>
                {v.toFixed(3)}
              </div>
              <div style={{ fontSize: 11, color: '#9ca3af' }}>target ≥ {m.target}</div>
            </div>
          );
        })}
      </div>

      <h2 style={{ marginTop: 32 }}>Run History (latest {runs.length})</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 12 }}>
        <thead>
          <tr style={{ background: '#f9fafb' }}>
            <th style={th}>Run ID</th>
            <th style={th}>Started</th>
            <th style={th}>Pass Rate</th>
            <th style={th}>Faith</th>
            <th style={th}>Rel</th>
            <th style={th}>Prec</th>
            <th style={th}>Rec</th>
            <th style={th}>Correct</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.run_id} style={{ borderBottom: '1px solid #f3f4f6' }}>
              <td style={td}>{r.run_id.slice(0, 24)}</td>
              <td style={td}>{new Date(r.started_at).toLocaleString()}</td>
              <td style={td}>
                {((r.passed_cases / Math.max(1, r.total_cases)) * 100).toFixed(0)}%
              </td>
              <td style={td}>{r.avg_faithfulness.toFixed(3)}</td>
              <td style={td}>{r.avg_answer_relevancy.toFixed(3)}</td>
              <td style={td}>{r.avg_context_precision.toFixed(3)}</td>
              <td style={td}>{r.avg_context_recall.toFixed(3)}</td>
              <td style={td}>{r.avg_answer_correctness.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const th: React.CSSProperties = { textAlign: 'left', padding: 8, fontSize: 12, color: '#6b7280' };
const td: React.CSSProperties = { padding: 8, fontSize: 12 };

function Stat({ label, value }: { label: string; value: any }) {
  return (
    <div style={{ flex: 1, padding: 16, background: '#f9fafb', borderRadius: 4 }}>
      <div style={{ fontSize: 12, color: '#6b7280' }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 600, marginTop: 4 }}>{value}</div>
    </div>
  );
}
