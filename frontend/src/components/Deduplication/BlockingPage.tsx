import React, { useEffect, useState } from 'react';
import { DownloadIcon, PlayIcon } from 'lucide-react';

// UI components (replace with your design system if needed)
// import { Card, Tabs, Tab, Button, Slider, Select, MultiSelect, Input, Heading } from 'your-ui-lib';

import { useAuth } from '../../context/AuthContext';
import { useSearchParams } from 'react-router-dom';

interface Dataset {
  id: number;
  filename: string;
}

const BlockingPage: React.FC = () => {
  const { token } = useAuth();
  const [searchParams] = useSearchParams();
  const urlSessionId = searchParams.get('session_id');
  const storedSessionId = typeof window !== 'undefined' ? localStorage.getItem('active_session_id') : null;
  const sessionId = urlSessionId || storedSessionId || null;

  // States for input dataset
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [loadingDatasets, setLoadingDatasets] = useState<boolean>(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // Fetch datasets on mount (like PreprocessingPage)
  useEffect(() => {
    if (!token) return;
    setLoadingDatasets(true);
    fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/datasets`, {
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${token}`
      }
    })
      .then(async res => {
        if (!res.ok) throw new Error('Failed to load datasets');
        const data = await res.json();
        setDatasets(data);
        setDatasetsError(null);
      })
      .catch(() => {
        setDatasets([]);
        setDatasetsError('Failed to load datasets');
      })
      .finally(() => setLoadingDatasets(false));
  }, [token]);

  // States for blocking method
  const [blockingMethod, setBlockingMethod] = useState<'minhash' | 'simhash'>('minhash');
  const [minhashPermutations, setMinhashPermutations] = useState<number>(128);
  const [minhashThreshold, setMinhashThreshold] = useState<number>(0.5);
  const [simhashBits, setSimhashBits] = useState<number>(64);
  const [simhashThreshold, setSimhashThreshold] = useState<number>(3);
  const [downloadName, setDownloadName] = useState<string>('');

  // States for columns
  const [columns, setColumns] = useState<string[]>([]);
  const [selectedFields, setSelectedFields] = useState<string[]>([]);

  // States for results and actions
  const [loadingColumns, setLoadingColumns] = useState<boolean>(false);
  const [loadingBlocking, setLoadingBlocking] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [summary, setSummary] = useState<Record<string, any> | null>(null);
  const [candidatePairsPath, setCandidatePairsPath] = useState<string | null>(null);
  const [previewPairs, setPreviewPairs] = useState<any[]>([]);
  const [sessionStepId, setSessionStepId] = useState<string | null>(null);

  const handleDownload = async () => {
    if (!candidatePairsPath || !token) return;
    try {
      const url = candidatePairsPath.startsWith('http')
        ? candidatePairsPath
        : `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${candidatePairsPath.replace(/^[.]/, '')}`;
      const response = await fetch(url, { headers: { 'Authorization': `Bearer ${token}` } });
      if (!response.ok) throw new Error('Download failed');
      const blob = await response.blob();
      const filename = `${downloadName || 'candidate_pairs'}.csv`;
      const blobUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(blobUrl);
    } catch (err) {
      console.error(err);
    }
  };

  // Fetch columns when dataset changes
  useEffect(() => {
    // If user uploaded a file, skip backend fetch
    if (uploadedFile) return;
    if (!selectedDataset || !token) {
      setColumns([]);
      return;
    }
    const fetchCols = async () => {
      setLoadingColumns(true);
      try {
        const res = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/datasets/${selectedDataset}/columns`, {
          headers: {
            'Accept': 'application/json',
            'Authorization': `Bearer ${token}`
          }
        });
        if (!res.ok) throw new Error('Failed to load columns');
        const cols = await res.json();
        setColumns(cols);
        setSelectedFields([]);
        setErrorMsg(null);
      } catch (e) {
        setColumns([]);
        setErrorMsg(e instanceof Error ? e.message : 'Failed to load columns');
      } finally {
        setLoadingColumns(false);
      }
    };
    fetchCols();
  }, [selectedDataset, token]);

  // File upload handler
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setUploadedFile(file);
      // Reset dataset selection so we know we're using the file
      setSelectedDataset('');
      // Parse first line to get columns
      const reader = new FileReader();
      reader.onload = evt => {
        const text = evt.target?.result as string;
        if (!text) return;
        const firstLine = text.split(/\r?\n/)[0];
        const cols = firstLine.split(',').map(c => c.trim()).filter(Boolean);
        setColumns(cols);
        setSelectedFields([]);
      };
      reader.readAsText(file);
    }
  };

  // Execute blocking through API
  const handleRunBlocking = async () => {
    if (!token) return;
    if (selectedFields.length === 0) {
      setErrorMsg('Please select at least one key field.');
      return;
    }

    setLoadingBlocking(true);
    setErrorMsg(null);
    setSummary(null);
    setPreviewPairs([]);
    setCandidatePairsPath(null);
    setSessionStepId(null);

    try {
      let datasetId: number | null = selectedDataset ? Number(selectedDataset) : null;

      // If the user uploaded a file and we don't yet have a dataset ID, upload it first
      if (!datasetId && uploadedFile) {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        formData.append('file_type', 'csv');
        const uploadRes = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/upload`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`
          },
          body: formData
        });
        if (!uploadRes.ok) {
          const err = await uploadRes.json().catch(() => ({}));
          throw new Error(err.detail || 'File upload failed');
        }
        const uploadData = await uploadRes.json();
        datasetId = uploadData.dataset_id;
        // Store so that subsequent operations treat it like a normal dataset
        setSelectedDataset(datasetId.toString());
        setDatasets(prev => [...prev, { id: datasetId!, filename: uploadedFile.name }]);
      }

      if (!datasetId) {
        throw new Error('No dataset selected or uploaded.');
      }

      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const sessionQS = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
      const body = {
        dataset_id: datasetId,
        method: blockingMethod === 'minhash' ? 'minhash_lsh' : 'simhash',
        key_fields: selectedFields,
        params: blockingMethod === 'minhash' ? {
          threshold: minhashThreshold,
          num_perm: minhashPermutations
        } : {
          threshold: simhashThreshold,
          num_bits: simhashBits
        }
      };

      const res = await fetch(`${API_URL}/api/v1/deduplication/pipeline/blocking${sessionQS}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(body)
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || 'Blocking failed');
      }

      const data = await res.json();
      if (data && typeof data.session_step_id !== 'undefined') {
        setSessionStepId(data.session_step_id || null);
      }
      if (data.status !== 'success') throw new Error(data.message || 'Blocking failed');
      setSummary(data.summary);
      const fullPath = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${data.candidate_pairs_csv_path || data.candidate_pairs_json_path}`;
      setCandidatePairsPath(fullPath);
      setPreviewPairs(data.preview || []);
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : 'Blocking failed');
    } finally {
      setLoadingBlocking(false);
    }
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-2xl font-bold mb-6 text-center">Blocking (Candidate Pair Generation)</h1>
      <div className="text-sm text-gray-600 text-right mb-4">
        {sessionId && (
          <span>
            Active session: <span className="font-mono">{sessionId}</span>
          </span>
        )}
        {sessionStepId && (
          <span>
            {sessionId ? ' • ' : ''}Last step: <span className="font-mono">{sessionStepId}</span>
          </span>
        )}
      </div>

      {/* Input Data */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Input Data</h2>

        <label className="block mb-2 font-medium">Use an existing dataset</label>
        <select
          className="w-full p-2 border rounded mb-4"
          value={selectedDataset}
          onChange={e => setSelectedDataset(e.target.value)}
          disabled={loadingDatasets || datasets.length === 0}
        >
          {loadingDatasets && <option value="">Loading datasets...</option>}
          {!loadingDatasets && !datasets.length && <option value="">No datasets found</option>}
          {!loadingDatasets && datasets.length > 0 && (
            <>
              <option value="">Select a dataset</option>
              {datasets.map(ds => (
                <option key={ds.id} value={ds.id}>{ds.filename}</option>
              ))}
            </>
          )}
        </select>
        {datasetsError && <div className="text-red-600 text-sm mb-2">{datasetsError}</div>}
        <label className="block mb-2 font-medium">Or upload preprocessed data</label>
        <input
          type="file"
          accept=".csv"
          className="block w-full mb-2"
          onChange={handleFileUpload}
        />
      </div>

      {/* Blocking Configuration */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Blocking Method</h2>
        <div className="mb-4">
          <div className="flex gap-4 mb-4">
            <button
              className={`px-4 py-2 rounded ${blockingMethod === 'minhash' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'}`}
              onClick={() => setBlockingMethod('minhash')}
            >
              MinHash LSH
            </button>
            <button
              className={`px-4 py-2 rounded ${blockingMethod === 'simhash' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'}`}
              onClick={() => setBlockingMethod('simhash')}
            >
              SimHash
            </button>
          </div>
          {blockingMethod === 'minhash' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Number of Permutations</label>
                <input
                  type="number"
                  className="w-full p-2 border rounded"
                  min={16}
                  max={512}
                  value={minhashPermutations}
                  onChange={e => setMinhashPermutations(Number(e.target.value))}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Threshold</label>
                <input
                  type="range"
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={minhashThreshold}
                  onChange={e => setMinhashThreshold(Number(e.target.value))}
                  className="w-full"
                />
                <span>{minhashThreshold}</span>
              </div>
            </div>
          )}
          {blockingMethod === 'simhash' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Number of Bits</label>
                <input
                  type="number"
                  className="w-full p-2 border rounded"
                  min={8}
                  max={256}
                  value={simhashBits}
                  onChange={e => setSimhashBits(Number(e.target.value))}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Similarity Threshold (Hamming Distance)</label>
                <input
                  type="range"
                  min={1}
                  max={10}
                  step={1}
                  value={simhashThreshold}
                  onChange={e => setSimhashThreshold(Number(e.target.value))}
                  className="w-full"
                />
                <span>{simhashThreshold}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Key Columns Selection */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Key Columns Selection</h2>
        <label className="block text-sm font-medium mb-1">Fields to use for blocking</label>
        <select
          multiple
          className="w-full p-2 border rounded h-28"
          value={selectedFields}
          onChange={e => setSelectedFields(Array.from(e.target.selectedOptions, opt => opt.value))}
          disabled={loadingColumns || columns.length === 0}
        >
          {columns.map(col => (
            <option key={col} value={col}>{col}</option>
          ))}
        </select>
        {loadingColumns && <div className="text-blue-600 text-sm mt-2">Loading columns...</div>}
        {errorMsg && <div className="text-red-600 text-sm mt-2">{errorMsg}</div>}
      </div>

      {/* Actions */}
      <div className="mb-6 flex flex-wrap gap-4">
        {/* Run */}
        <button
          className={`px-4 py-2 rounded text-white font-semibold flex items-center gap-2 ${loadingBlocking ? 'bg-blue-300' : 'bg-blue-500 hover:bg-blue-600'}`}
          onClick={handleRunBlocking}
          disabled={loadingBlocking}
        >
          <PlayIcon size={16}/> {loadingBlocking ? 'Running…' : 'Run Blocking'}
        </button>
      </div>

      {/* Result */}
      {summary && (
        <div className="my-6 bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-3">Blocking Summary</h3>
          {/* Download section */}
          {candidatePairsPath && (
            <div className="mb-4 flex items-center gap-2">
              <input
                type="text"
                className="flex-1 p-2 border rounded"
                placeholder="Custom filename (optional)"
                value={downloadName}
                onChange={e => setDownloadName(e.target.value)}
              />
              <button
                onClick={handleDownload}
                className="px-4 py-2 rounded text-white bg-gray-600 hover:bg-gray-700 font-semibold flex items-center gap-2"
              >
                <DownloadIcon size={16}/> Download Candidate Pairs
              </button>
            </div>
          )}
          <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
            <li><span className="font-medium">Method:</span> {summary.method}</li>
            <li><span className="font-medium">Key Fields:</span> {Array.isArray(summary.key_fields) ? summary.key_fields.join(', ') : ''}</li>
            <li><span className="font-medium">Total Records:</span> {summary.total_records}</li>
            <li><span className="font-medium">Candidate Pairs:</span> {summary.candidate_pairs}</li>
            <li><span className="font-medium">Reduction Ratio:</span> {(summary.reduction_ratio * 100).toFixed(2)}%</li>
          </ul>
        </div>
      )}

      {previewPairs.length > 0 && (
        <div className="my-6 bg-white p-4 rounded-lg shadow overflow-x-auto">
          <h3 className="text-lg font-semibold mb-3">Candidate Pair Preview (first {previewPairs.length})</h3>
          <table className="min-w-full text-sm border">
            <thead>
              <tr className="bg-gray-100">
                <th className="border px-2 py-1">Pair ID</th>
                <th className="border px-2 py-1">Record 1 ID</th>
                <th className="border px-2 py-1">Record 2 ID</th>
              </tr>
            </thead>
            <tbody>
              {previewPairs.map(p => (
                <tr key={p.pair_id} className="text-center">
                  <td className="border px-2 py-1">{p.pair_id}</td>
                  <td className="border px-2 py-1">{p.record1_id}</td>
                  <td className="border px-2 py-1">{p.record2_id}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default BlockingPage;
