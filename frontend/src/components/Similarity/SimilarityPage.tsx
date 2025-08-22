import React, { useEffect, useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useSearchParams } from 'react-router-dom';

interface Dataset { id: number; filename: string; }

interface FieldConfig {
  field: string;
  metric: string;
  weight: number;
}

const textMetrics = ['Jaro-Winkler', 'TF-IDF + Cosine'];
const categoricalMetrics = ['Exact Match', 'Jaccard Similarity'];

const SimilarityPage: React.FC = () => {
  const { token } = useAuth();
  const [searchParams] = useSearchParams();
  const urlSessionId = searchParams.get('session_id');
  const storedSessionId = typeof window !== 'undefined' ? localStorage.getItem('active_session_id') : null;
  const sessionId = urlSessionId || storedSessionId || null;
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);

  const [columnNames, setColumnNames] = useState<string[]>([]);
  const [loadingColumns, setLoadingColumns] = useState(false);
  const [columnsError, setColumnsError] = useState<string | null>(null);

  // candidatePairsPath kept for backend compatibility but no user input UI
  const [candidatePairsPath, setCandidatePairsPath] = useState<string>('');

  // Config states
  const [textFieldConfigs, setTextFieldConfigs] = useState<FieldConfig[]>([]);
  const [numericFieldConfigs, setNumericFieldConfigs] = useState<FieldConfig[]>([]);
  const [categoricalFieldConfigs, setCategoricalFieldConfigs] = useState<FieldConfig[]>([]);
  const [compositeThreshold, setCompositeThreshold] = useState<number>(0.7);

  // Actions and results
  const [loadingCalc, setLoadingCalc] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [summary, setSummary] = useState<Record<string, any> | null>(null);
  const [similarityPath, setSimilarityPath] = useState<string | null>(null);
  const [downloadName, setDownloadName] = useState<string>('');
  const [sessionStepId, setSessionStepId] = useState<string | null>(null);

  // Fetch datasets
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

  // Fetch columns for selected dataset
  useEffect(() => {
    if (!selectedDataset || !token) {
      setColumnNames([]);
      return;
    }
    setLoadingColumns(true);
    setColumnsError(null);
    fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/datasets/${selectedDataset}/columns`, {
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${token}`
      }
    })
      .then(async res => {
        if (!res.ok) throw new Error('Failed to load columns');
        const data = await res.json();
        const cols = Array.isArray(data) ? data : (data.columns || []);
        setColumnNames(cols);
        setColumnsError(null);
      })
      .catch(() => {
        setColumnNames([]);
        setColumnsError('Failed to load columns');
      })
      .finally(() => setLoadingColumns(false));
  }, [selectedDataset, token]);

  // Weight sum for warning
  const totalWeight = [
    ...textFieldConfigs,
    ...numericFieldConfigs,
    ...categoricalFieldConfigs,
  ].reduce((sum, c) => sum + c.weight, 0);

  // Handlers for config lists
  const addTextConfig = () => setTextFieldConfigs([...textFieldConfigs, { field: '', metric: textMetrics[0], weight: 0.2 }]);
  const addNumericConfig = () => setNumericFieldConfigs([...numericFieldConfigs, { field: '', metric: 'Normalized Distance', weight: 0.2 }]);
  const addCategoricalConfig = () => setCategoricalFieldConfigs([...categoricalFieldConfigs, { field: '', metric: categoricalMetrics[0], weight: 0.2 }]);

  const updateConfig = (list: FieldConfig[], setList: React.Dispatch<React.SetStateAction<FieldConfig[]>>, idx: number, key: keyof FieldConfig, value: any) => {
    const copy = [...list];
    copy[idx] = { ...copy[idx], [key]: value };
    setList(copy);
  };
  const removeConfig = (list: FieldConfig[], setList: React.Dispatch<React.SetStateAction<FieldConfig[]>>, idx: number) => {
    setList(list.filter((_, i) => i !== idx));
  };

  // Actions
  const buildFieldConfigs = () => {
    const cfg: Record<string, any> = {};
    textFieldConfigs.forEach(c => {
      if (c.field) cfg[c.field] = { type: 'text', method: c.metric === 'Jaro-Winkler' ? 'jaro_winkler' : 'tfidf_cosine', weight: c.weight };
    });
    numericFieldConfigs.forEach(c => {
      if (c.field) cfg[c.field] = { type: 'numeric', method: 'normalized_distance', weight: c.weight };
    });
    categoricalFieldConfigs.forEach(c => {
      if (c.field) cfg[c.field] = { type: 'categorical', method: c.metric === 'Exact Match' ? 'exact_match' : 'jaccard', weight: c.weight };
    });
    return cfg;
  };

  const handleSimilarityCalc = async () => {
    if (!token) return;
    if (!candidatePairsPath) {
      setErrorMsg('Candidate pairs path is required');
      return;
    }
    setLoadingCalc(true);
    setErrorMsg(null);
    setSummary(null);
    setSimilarityPath(null);
    setSessionStepId(null);
    try {
      const payload = {
        dataset_id: Number(selectedDataset),
        candidate_pairs_path: candidatePairsPath,
        field_configs: buildFieldConfigs(),
        threshold: compositeThreshold,
      };
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const sessionQS = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
      const res = await fetch(`${API_URL}/api/v1/deduplication/pipeline/similarity${sessionQS}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error('Similarity calculation failed');
      const data = await res.json();
      if (data && typeof data.session_step_id !== 'undefined') {
        setSessionStepId(data.session_step_id || null);
      }
      setSummary(data.summary || null);
      setSimilarityPath(data.similarity_results_path || null);
    } catch (err: any) {
      setErrorMsg(err.message || 'Error');
    } finally {
      setLoadingCalc(false);
    }
  };

  const handleDownload = async () => {
    if (!similarityPath || !token) return;
    try {
      const url = similarityPath.startsWith('http') ? similarityPath : `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${similarityPath.replace(/^[.]/,'')}`;
      const resp = await fetch(url, { headers: { 'Authorization': `Bearer ${token}` } });
      if (!resp.ok) throw new Error('Download failed');
      const blob = await resp.blob();
      const extMatch = similarityPath.match(/\.\w+$/);
      const ext = extMatch ? extMatch[0] : '.json';
      const filename = `${downloadName || 'similarity_results'}${ext}`;
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

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-2xl font-bold mb-6 text-center">Similarity Calculation</h1>
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
        <label className="block mb-2 font-medium">Use the original dataset</label>
        <select
          className="w-full p-2 border rounded mb-4"
          value={selectedDataset}
          onChange={e => setSelectedDataset(e.target.value)}
          disabled={loadingDatasets || datasets.length === 0}
        >
          {loadingDatasets && <option>Loading datasets...</option>}
          {!loadingDatasets && datasets.length === 0 && <option>No datasets found</option>}
          {!loadingDatasets && datasets.length > 0 && (
            <>
              <option value="">Select dataset...</option>
              {datasets.map(ds => (
                <option key={ds.id} value={ds.id}>{ds.filename}</option>
              ))}
            </>
          )}
        </select>
        {datasetsError && <div className="text-red-600 text-sm mb-2">{datasetsError}</div>}
        <label className="block mb-2 font-medium">Upload candidate pairs file (CSV or JSON)</label>
        <input
          type="file"
          accept=".csv,.json"
          className="block w-full mb-2"
          onChange={async e => {
            if (e.target.files && e.target.files.length > 0) {
              const file = e.target.files[0];
              try {
                const form = new FormData();
                form.append('file', file);
                const uploadRes = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/artifacts/upload`, {
                  method: 'POST',
                  headers: {
                    'Authorization': `Bearer ${token}`
                  },
                  body: form
                });
                if (!uploadRes.ok) throw new Error('Upload failed');
                const upData = await uploadRes.json();
                setCandidatePairsPath(upData.artifact_path);
                setErrorMsg(null);
              } catch (err: any) {
                setErrorMsg(err.message || 'Upload error');
              }
            }
          }}
        />
      </div>

      {/* Similarity Metrics Configuration */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Similarity Metrics Configuration</h2>
        {/* Text Fields */}
        <h3 className="font-semibold mb-2">Text Fields</h3>
        {textFieldConfigs.map((cfg, idx) => (
          <div key={idx} className="flex gap-2 items-center mb-2">
            <select
              className="p-2 border rounded"
              value={cfg.field}
              onChange={e => updateConfig(textFieldConfigs, setTextFieldConfigs, idx, 'field', e.target.value)}
            >
              <option value="">Select field...</option>
              {columnNames.map(col => <option key={col} value={col}>{col}</option>)}
            </select>
            <select
              className="p-2 border rounded"
              value={cfg.metric}
              onChange={e => updateConfig(textFieldConfigs, setTextFieldConfigs, idx, 'metric', e.target.value)}
            >
              {textMetrics.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={cfg.weight}
              onChange={e => updateConfig(textFieldConfigs, setTextFieldConfigs, idx, 'weight', parseFloat(e.target.value))}
            />
            <span>{cfg.weight}</span>
            <button className="text-red-600 text-xl" onClick={() => removeConfig(textFieldConfigs, setTextFieldConfigs, idx)} title="Remove">🗑️</button>
          </div>
        ))}
        <button className="text-blue-600 font-semibold mb-2" onClick={addTextConfig}>+ Add text field</button>

        {/* Numeric Fields */}
        <h3 className="font-semibold mb-2 mt-4">Numeric Fields</h3>
        {numericFieldConfigs.map((cfg, idx) => (
          <div key={idx} className="flex gap-2 items-center mb-2">
            <select
              className="p-2 border rounded"
              value={cfg.field}
              onChange={e => updateConfig(numericFieldConfigs, setNumericFieldConfigs, idx, 'field', e.target.value)}
            >
              <option value="">Select field...</option>
              {columnNames.map(col => <option key={col} value={col}>{col}</option>)}
            </select>
            <span>Normalized Distance</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={cfg.weight}
              onChange={e => updateConfig(numericFieldConfigs, setNumericFieldConfigs, idx, 'weight', parseFloat(e.target.value))}
            />
            <span>{cfg.weight}</span>
            <button className="text-red-600 text-xl" onClick={() => removeConfig(numericFieldConfigs, setNumericFieldConfigs, idx)} title="Remove">🗑️</button>
          </div>
        ))}
        <button className="text-blue-600 font-semibold mb-2" onClick={addNumericConfig}>+ Add numeric field</button>

        {/* Categorical Fields */}
        <h3 className="font-semibold mb-2 mt-4">Categorical Fields</h3>
        {categoricalFieldConfigs.map((cfg, idx) => (
          <div key={idx} className="flex gap-2 items-center mb-2">
            <select
              className="p-2 border rounded"
              value={cfg.field}
              onChange={e => updateConfig(categoricalFieldConfigs, setCategoricalFieldConfigs, idx, 'field', e.target.value)}
            >
              <option value="">Select field...</option>
              {columnNames.map(col => <option key={col} value={col}>{col}</option>)}
            </select>
            <select
              className="p-2 border rounded"
              value={cfg.metric}
              onChange={e => updateConfig(categoricalFieldConfigs, setCategoricalFieldConfigs, idx, 'metric', e.target.value)}
            >
              {categoricalMetrics.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={cfg.weight}
              onChange={e => updateConfig(categoricalFieldConfigs, setCategoricalFieldConfigs, idx, 'weight', parseFloat(e.target.value))}
            />
            <span>{cfg.weight}</span>
            <button className="text-red-600 text-xl" onClick={() => removeConfig(categoricalFieldConfigs, setCategoricalFieldConfigs, idx)} title="Remove">🗑️</button>
          </div>
        ))}
        <button className="text-blue-600 font-semibold mb-2" onClick={addCategoricalConfig}>+ Add categorical field</button>

        {/* Weight sum warning */}
        <div className="mt-4">
          <span className="font-semibold">Total weight: {totalWeight.toFixed(2)}</span>
          {totalWeight !== 1 && <span className="ml-2 text-red-600">(Warning: total weight should be 1)</span>}
        </div>
      </div>

    {/* Composite Similarity Threshold */}
    <div className="mb-6 bg-white p-4 rounded-lg shadow">
      <h2 className="text-lg font-semibold mb-3">Composite Similarity Threshold</h2>
      <label className="block mb-2 font-medium">Decision threshold</label>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={compositeThreshold}
        onChange={e => setCompositeThreshold(parseFloat(e.target.value))}
        className="w-full"
      />
      <span>{compositeThreshold}</span>
    </div>

    {/* Actions */}
    <div className="mb-6 flex flex-wrap gap-4">
      <button
        className="px-4 py-2 rounded text-white bg-blue-500 hover:bg-blue-600 font-semibold flex items-center gap-2 disabled:opacity-50"
        onClick={handleSimilarityCalc}
        disabled={loadingCalc}
      >
        {loadingCalc ? 'Processing...' : 'Run Similarity'}
      </button>
      {summary && (
        <>
          <input
            type="text"
            className="p-2 border rounded"
            placeholder="Custom filename (optional)"
            value={downloadName}
            onChange={e => setDownloadName(e.target.value)}
          />
          <button
            className="px-4 py-2 rounded text-white bg-gray-600 hover:bg-gray-700 font-semibold flex items-center gap-2"
            onClick={handleDownload}
          >
            Download Results
          </button>
        </>
      )}
    </div>

    {summary && (
      <div className="my-6 bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-3">Similarity Summary</h3>
        <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
          <li><span className="font-medium">Candidate Pairs:</span> {summary.candidate_pairs}</li>
          <li><span className="font-medium">Similar Pairs:</span> {summary.similar_pairs}</li>
          <li><span className="font-medium">Threshold:</span> {summary.threshold}</li>
        </ul>
      </div>
    )}
    {errorMsg && <div className="text-red-600 text-sm mt-2">{errorMsg}</div>}
  </div>
);
};

export default SimilarityPage;
