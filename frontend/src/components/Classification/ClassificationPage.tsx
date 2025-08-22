import React, { useEffect, useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useSearchParams } from 'react-router-dom';

interface Dataset { id: number; filename: string; }

type ClassificationMethod = 'rf' | 'xgb' | 'siamese';

const ClassificationPage: React.FC = () => {
  const { token } = useAuth();
  const [searchParams] = useSearchParams();
  const urlSessionId = searchParams.get('session_id');
  const storedSessionId = typeof window !== 'undefined' ? localStorage.getItem('active_session_id') : null;
  const sessionId = urlSessionId || storedSessionId || null;

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);

  const [similarityResultsFile, setSimilarityResultsFile] = useState<File | null>(null);

  // UI & backend states
  const [loadingClassify, setLoadingClassify] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [summary, setSummary] = useState<any | null>(null);
  const [resultsPath, setResultsPath] = useState<string | null>(null);
  const [sessionStepId, setSessionStepId] = useState<string | null>(null);

  // Classification method and params
  const [classificationMethod, setClassificationMethod] = useState<ClassificationMethod>('rf');
  const [rfEstimators, setRfEstimators] = useState<number>(100);
  const [rfMaxDepth, setRfMaxDepth] = useState<number>(10);
  const [xgbLearningRate, setXgbLearningRate] = useState<number>(0.1);
  const [xgbEstimators, setXgbEstimators] = useState<number>(100);
  const [siameseLearningRate, setSiameseLearningRate] = useState<number>(0.001);

  // Classification params
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.8);
  const [showFeatureImportance, setShowFeatureImportance] = useState<boolean>(false);

  // Actions/results
  const [classificationDone, setClassificationDone] = useState(false);
  const [duplicatePairs, setDuplicatePairs] = useState<any[]>([]);
  const [featureImportance, setFeatureImportance] = useState<any[] | null>(null);
  // filename for download
  const [downloadName, setDownloadName] = useState<string>('');

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
        // Do not preselect any dataset; user must choose explicitly.
        setDatasetsError(null);
      })
      .catch(() => {
        setDatasets([]);
        setDatasetsError('Failed to load datasets');
      })
      .finally(() => setLoadingDatasets(false));
  }, [token]);

  // Clear error message when required inputs become valid
  useEffect(() => {
    if (selectedDataset && similarityResultsFile) {
      setErrorMsg(null);
    }
  }, [selectedDataset, similarityResultsFile]);

    // Map short code to backend method name
  const methodMap: Record<ClassificationMethod, string> = {
    rf: 'random_forest',
    xgb: 'xgboost',
    siamese: 'siamese_network'
  };

  // Build params depending on chosen method
  const buildParams = () => {
    const base: any = { confidence_threshold: confidenceThreshold };
    if (classificationMethod === 'rf') {
      base.n_estimators = rfEstimators;
      base.max_depth = rfMaxDepth;
    } else if (classificationMethod === 'xgb') {
      base.learning_rate = xgbLearningRate;
      base.n_estimators = xgbEstimators;
    } else if (classificationMethod === 'siamese') {
      base.learning_rate = siameseLearningRate;
    }
    return base;
  };

  // Download classification results file
  const handleDownload = async () => {
    if (!resultsPath || !token) return;
    try {
      const url = resultsPath.startsWith('http') ? resultsPath : `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${resultsPath.replace(/^[.]/, '')}`;
      const resp = await fetch(url, { headers: { 'Authorization': `Bearer ${token}` } });
      if (!resp.ok) throw new Error('Download failed');
      const blob = await resp.blob();
      const extMatch = resultsPath.match(/\.\w+$/);
      const ext = extMatch ? extMatch[0] : '.json';
      const filename = `${downloadName || 'classification_results'}${ext}`;
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

  // Classification action – real backend call
  const handleClassification = async () => {
    if (!token) return;
    if (!selectedDataset) { setErrorMsg('Please select a dataset'); return; }
    if (!similarityResultsFile) { setErrorMsg('Please upload a similarity file'); return; }
    try {
      setLoadingClassify(true);
      setErrorMsg(null);
      setSummary(null);
      setDuplicatePairs([]);
      setSessionStepId(null);

      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const sessionQS = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';

      // 1. Upload similarity results file to get path
      const form = new FormData();
      form.append('file', similarityResultsFile);
      const upRes = await fetch(`${API_URL}/api/v1/artifacts/upload`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: form
      });
      if (!upRes.ok) throw new Error('File upload failed');
      const upData = await upRes.json();
      const similarityPath = upData.artifact_path;

      // 2. Build payload and call classification endpoint
      const payload = {
        dataset_id: Number(selectedDataset),
        similarity_results_path: similarityPath,
        method: methodMap[classificationMethod],
        params: buildParams()
      };
      const res = await fetch(`${API_URL}/api/v1/deduplication/pipeline/classification${sessionQS} `, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Classification error: ${errText || res.status}`);
      }
      const data = await res.json();
      console.log('Classification response', data);
      if (data && typeof data.session_step_id !== 'undefined') {
        setSessionStepId(data.session_step_id || null);
      }
      if (data.status === 'error' || !data.summary) {
        setErrorMsg(data.message || data.error || 'Classification failed');
        setClassificationDone(false);
        return;
      }
      setSummary(data.summary);
      setDuplicatePairs(data.preview || []);
      setResultsPath(data.classification_results_path || null);
      if (showFeatureImportance && data.summary?.feature_importance) {
        setFeatureImportance(data.summary.feature_importance);
      } else {
        setFeatureImportance(null);
      }
      setClassificationDone(true);
    } catch (err: any) {
      setErrorMsg(err.message || 'Unknown error');
    } finally {
      setLoadingClassify(false);
    }
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-2xl font-bold mb-6 text-center">Classification des Doublons</h1>
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
        <label className="block mb-2 font-medium">Select Dataset</label>
        <select
          className="w-full p-2 border rounded mb-4"
          value={selectedDataset}
          onChange={e => setSelectedDataset(e.target.value)}
          disabled={loadingDatasets || datasets.length === 0}
        >
          {/* Placeholder */}
          <option value="" disabled hidden>Select a dataset...</option>
          {loadingDatasets && <option>Loading...</option>}
          {!loadingDatasets && datasets.length === 0 && <option>No dataset</option>}
          {!loadingDatasets && datasets.length > 0 && datasets.map(ds => (
            <option key={ds.id} value={ds.id}>{ds.filename}</option>
          ))}
        </select>
        {datasetsError && <div className="text-red-600 text-sm mb-2">{datasetsError}</div>}
        <label className="block mb-2 font-medium">Upload Similarity Results</label>
        <input
          type="file"
          accept=".csv,.json"
          className="block w-full mb-2"
          onChange={e => {
            const file = e.target.files?.[0] || null;
            setSimilarityResultsFile(file);
          }}
        />
        {similarityResultsFile && (
           <div className="text-sm text-gray-600 mb-2">Selected file: {similarityResultsFile.name}</div>
         )}
      </div>

      {/* Classification Method */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Classification Method</h2>
        <div className="flex gap-4 mb-4">
          <button
            className={`px-4 py-2 rounded font-semibold border ${classificationMethod === 'rf' ? 'bg-blue-500 text-white' : 'bg-white text-blue-500 border-blue-500'}`}
            onClick={() => setClassificationMethod('rf')}
          >
            Random Forest
          </button>
          <button
            className={`px-4 py-2 rounded font-semibold border ${classificationMethod === 'xgb' ? 'bg-blue-500 text-white' : 'bg-white text-blue-500 border-blue-500'}`}
            onClick={() => setClassificationMethod('xgb')}
          >
            XGBoost
          </button>
          <button
            className={`px-4 py-2 rounded font-semibold border ${classificationMethod === 'siamese' ? 'bg-blue-500 text-white' : 'bg-white text-blue-500 border-blue-500'}`}
            onClick={() => setClassificationMethod('siamese')}
          >
            Siamese Network
          </button>
        </div>
        {/* Method-specific params */}
        {classificationMethod === 'rf' && (
          <div className="flex flex-col gap-2 mb-2">
            <label className="font-medium">Number of Estimators (trees)</label>
            <input
              type="number"
              className="p-2 border rounded"
              value={rfEstimators}
              min={1}
              onChange={e => setRfEstimators(Number(e.target.value))}
            />
            <label className="font-medium">Maximum Depth</label>
            <input
              type="number"
              className="p-2 border rounded"
              value={rfMaxDepth}
              min={1}
              onChange={e => setRfMaxDepth(Number(e.target.value))}
            />
          </div>
        )}
        {classificationMethod === 'xgb' && (
          <div className="flex flex-col gap-2 mb-2">
            <label className="font-medium">Taux d'Apprentissage (Learning Rate)</label>
            <input
              type="number"
              step="0.01"
              className="p-2 border rounded"
              value={xgbLearningRate}
              min={0}
              onChange={e => setXgbLearningRate(Number(e.target.value))}
            />
            <label className="font-medium">Nombre d'Estimateurs</label>
            <input
              type="number"
              className="p-2 border rounded"
              value={xgbEstimators}
              min={1}
              onChange={e => setXgbEstimators(Number(e.target.value))}
            />
          </div>
        )}
        {classificationMethod === 'siamese' && (
          <div className="flex flex-col gap-2 mb-2">
            <label className="font-medium">Taux d'Apprentissage (Learning Rate)</label>
            <input
              type="number"
              step="0.0001"
              className="p-2 border rounded"
              value={siameseLearningRate}
              min={0}
              onChange={e => setSiameseLearningRate(Number(e.target.value))}
            />
          </div>
        )}
      </div>

      {/* Classification Parameters */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Classification Parameters</h2>
        <label className="block mb-2 font-medium">Confidence Threshold</label>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={confidenceThreshold}
          onChange={e => setConfidenceThreshold(parseFloat(e.target.value))}
          className="w-full"
        />
        <span>{confidenceThreshold}</span>
        {(classificationMethod === 'rf' || classificationMethod === 'xgb') && (
          <div className="flex items-center mt-4">
            <label className="mr-2 font-medium">Show Feature Importance</label>
            <input
              type="checkbox"
              checked={showFeatureImportance}
              onChange={e => setShowFeatureImportance(e.target.checked)}
            />
          </div>
        )}
      </div>

      {errorMsg && (
        <div className="mb-4 p-2 bg-red-100 text-red-600 rounded">
          {errorMsg}
        </div>
      )}

      {/* Actions */}
      <div className="mb-6 flex flex-wrap gap-4">
        <button
          className={`px-4 py-2 rounded font-semibold ${(!selectedDataset || !similarityResultsFile) ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-600 text-white'}`}
          disabled={!selectedDataset || !similarityResultsFile || loadingClassify}
          onClick={handleClassification}
        >
          {loadingClassify ? 'Please wait...' : 'Run Classification'}
        </button>
        {classificationDone && (
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

      {/* Feature Importance Chart */}
      {featureImportance && (
        <div className="my-4 p-3 bg-yellow-50 border border-yellow-200 text-yellow-800 rounded">
          <h3 className="font-semibold mb-2">Importance des Caractéristiques</h3>
          <ul>
            {featureImportance.map((fi, idx) => (
              <li key={idx}>{fi.feature}: {fi.importance}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Summary & Preview */}
      {classificationDone && summary && (
        <div className="my-4 p-3 bg-blue-50 border border-blue-200 text-blue-800 rounded">
          <h3 className="font-semibold mb-2">Summary:</h3>
          <pre className="text-xs overflow-x-auto">{JSON.stringify(summary, null, 2)}</pre>
        </div>
      )}

      {classificationDone && duplicatePairs.length > 0 && (
        <div className="my-4 p-3 bg-green-50 border border-green-200 text-green-800 rounded">
          <h3 className="font-semibold mb-2">Duplicate Pairs Preview :</h3>
          <pre className="text-xs overflow-x-auto">{JSON.stringify(duplicatePairs, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default ClassificationPage;
