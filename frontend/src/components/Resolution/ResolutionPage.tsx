import React, { useEffect, useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useSearchParams } from 'react-router-dom';

interface Dataset { id: number; filename: string }

type ResolutionMethod = 'keep_first' | 'keep_most_complete' | 'merge';

const ResolutionPage: React.FC = () => {
  const { token } = useAuth();
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get('session_id') || localStorage.getItem('active_session_id') || '';

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [clusterFile, setClusterFile] = useState<File | null>(null);
  const [method, setMethod] = useState<ResolutionMethod>('keep_first');

  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [summary, setSummary] = useState<any | null>(null);
  const [cleanPath, setCleanPath] = useState<string | null>(null);
  const [outputName, setOutputName] = useState<string>('cleaned_dataset');
  const [sessionStepId, setSessionStepId] = useState<string | null>(null);

  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const normalizePath = (p?: string): string => {
    if (!p) return '';
    if (p.startsWith('http')) {
      if (!sessionId) return p;
      return p.includes('?') ? `${p}&session_id=${encodeURIComponent(sessionId)}` : `${p}?session_id=${encodeURIComponent(sessionId)}`;
    }
    const normalized = p.replace(/\\/g, '/').replace(/^\/+/, '');
    const url = `${API_URL}/${normalized}`;
    if (!sessionId) return url;
    return url.includes('?') ? `${url}&session_id=${encodeURIComponent(sessionId)}` : `${url}?session_id=${encodeURIComponent(sessionId)}`;
  };

  // fetch datasets on mount
  useEffect(() => {
    if (!token) return;
    (async () => {
      try {
        const res = await fetch(`${API_URL}/api/v1/datasets`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (!res.ok) throw new Error('Failed to fetch datasets');
        const data = await res.json();
        setDatasets(data);
      } catch (err: any) {
        setErrorMsg(err.message);
      }
    })();
  }, [token]);

  const handleRunResolution = async () => {
    if (!token) return;
    if (!selectedDataset) { setErrorMsg('Please select a dataset'); return; }
    if (!clusterFile) { setErrorMsg('Upload cluster assignments CSV'); return; }

    try {
      setLoading(true); setErrorMsg(null); setSummary(null);

      // Upload cluster file first
      const form = new FormData();
      form.append('file', clusterFile);
      const upRes = await fetch(`${API_URL}/api/v1/artifacts/upload${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`, {
        method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: form
      });
      if (!upRes.ok) throw new Error('Upload failed');
      const upData = await upRes.json();
      const clusterPath = upData.artifact_path || upData.file_path;

      // POST to resolution endpoint
      const payload = {
        dataset_id: Number(selectedDataset),
        clustering_results_path: clusterPath,
        method,
        params: {}
      };
      const res = await fetch(`${API_URL}/api/v1/deduplication/pipeline/resolution${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (typeof data.session_step_id !== 'undefined') {
        setSessionStepId(data.session_step_id || null);
      }
      if (!res.ok || data.status === 'error') {
        console.error('Resolution API error', data);
        throw new Error(data.detail || data.message || 'Resolution failed');
      }
      setSummary(data.summary);
      setCleanPath(data.resolved_dataset_path);
    } catch (err: any) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-2xl font-semibold mb-6">Results &amp; Resolution</h1>
      {sessionId && (
        <div className="mb-2 text-xs text-gray-500 text-center">
          Active session: <span className="font-mono">{sessionId}</span>
          {sessionStepId && (
            <>
              {' '}• Last step: <span className="font-mono">{sessionStepId}</span>
            </>
          )}
        </div>
      )}
      {/* Input section */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow grid gap-4">
        <div>
          <label className="block font-medium mb-1">Use Dataset</label>
          <select className="p-2 border rounded w-full" value={selectedDataset} onChange={e => setSelectedDataset(e.target.value)}>
            <option value="">-- Select Dataset --</option>
            {datasets.map(d => <option key={d.id} value={d.id}>{d.filename}</option>)}
          </select>
        </div>
        <div>
          <label className="block font-medium mb-1">Upload Cluster Assignments (.csv)</label>
          <input type="file" accept=".csv" onChange={e => setClusterFile(e.target.files?.[0] || null)} />
        </div>
        <div>
          <label className="block font-medium mb-1">Resolution Strategy</label>
          <select className="p-2 border rounded w-full" value={method} onChange={e => setMethod(e.target.value as ResolutionMethod)}>
            <option value="keep_first">Keep First Record</option>
            <option value="keep_most_complete">Keep Most Complete Record</option>
            <option value="merge">Merge Records</option>
          </select>
        </div>
        <div>
          <button disabled={loading} onClick={handleRunResolution} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-60">
            {loading ? 'Running…' : 'Run Resolution'}
          </button>
        </div>
      </div>

      {errorMsg && <div className="p-3 bg-red-100 text-red-700 rounded mb-4">{errorMsg}</div>}

      {/* Summary */}
      {summary && (
        <div className="mb-6 bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-3">Resolution Summary</h2>
          <ul className="list-disc list-inside text-sm space-y-1">
            <li><span className="font-medium">Method:</span> {summary.method}</li>
            <li><span className="font-medium">Total Clusters:</span> {summary.total_clusters}</li>
            <li><span className="font-medium">Records Kept:</span> {summary.records_kept}</li>
            <li><span className="font-medium">Records Removed:</span> {summary.records_removed}</li>
            {summary.records_removed && summary.records_kept && (
              <li><span className="font-medium">Reduction %:</span> {((summary.records_removed / (summary.records_kept + summary.records_removed)) * 100).toFixed(2)}%</li>
            )}
          </ul>
          {cleanPath && (
            <div className="flex flex-col sm:flex-row sm:items-center gap-3 mt-4">
              <input type="text" className="p-2 border rounded w-full sm:flex-grow" placeholder="Custom filename (optional)" value={outputName} onChange={e => setOutputName(e.target.value)} />
              <a href={normalizePath(cleanPath)} download={`${outputName || 'cleaned_dataset'}.csv`} className="px-4 py-2 bg-green-700 text-white rounded hover:bg-green-800 whitespace-nowrap text-center">
                Download Cleaned Dataset
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ResolutionPage;
