import React, { useState, useEffect } from 'react';
import MissingValueChart from '../components/Imputation/MissingValueChart';
import { useAuth } from '../context/AuthContext';
import { useSearchParams } from 'react-router-dom';

// Interfaces
interface Dataset {
  id: number;
  filename: string;
}

interface ImputationParams {
  simple: {
    strategy: 'mean' | 'median' | 'most_frequent';
  };
  mice: {
    n_imputations: number;
    max_iter: number;
    random_state?: number;
  };
  knn: {
    n_neighbors: number;
    weights: 'uniform' | 'distance';
  };
  missforest: {
    n_estimators: number;
    max_depth?: number;
    random_state?: number;
  };
}

type AlgorithmKey = keyof ImputationParams;

const algorithms = [
  { key: 'simple', label: 'Simple' },
  { key: 'mice', label: 'MICE' },
  { key: 'knn', label: 'kNN' },
  { key: 'missforest', label: 'MissForest' },
] as const;

const ImputationPage: React.FC = () => {
  const { token } = useAuth();
  const [searchParams] = useSearchParams();
  const sessionIdFromUrl = searchParams.get('session_id') || localStorage.getItem('active_session_id');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const DRAFT_KEY = 'imputationDraft';

  const defaultParams: ImputationParams = {
    simple: { strategy: 'mean' },
    mice: { n_imputations: 5, max_iter: 10, random_state: 42 },
    knn: { n_neighbors: 5, weights: 'uniform' },
    missforest: { n_estimators: 100, max_depth: 10, random_state: 42 },
  };

  const [selectedAlgo, setSelectedAlgo] = useState<AlgorithmKey>('simple');
  // Algorithm parameters
  const [params, setParams] = useState<ImputationParams>(defaultParams);
  
  // Loading states
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [imputationLoading, setImputationLoading] = useState<boolean>(false);
  const [imputationError, setImputationError] = useState<string | null>(null);
  const [imputationResult, setImputationResult] = useState<any>(null);
  // Keep history for potential analytics; not shown in UI
  const [runHistory, setRunHistory] = useState<any[]>([]);
  // Async job tracking
  const [taskId, setTaskId] = useState<string | null>(null);
  const [sessionStepId, setSessionStepId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);

  const [outputName, setOutputName] = useState<string>('');

  // Load draft on mount
  useEffect(() => {
    const stored = localStorage.getItem(DRAFT_KEY);
    if (stored) {
      try {
        const draft = JSON.parse(stored);
        setSelectedDataset(draft.selectedDataset || '');
        setSelectedAlgo(draft.selectedAlgo || 'simple');
        setParams(draft.params || defaultParams);
        setOutputName(draft.outputName || '');
      } catch (_) {
        /* ignore corrupted */
      }
    }
  }, []);

  // Persist draft whenever relevant state changes
  useEffect(() => {
    const draft = {
      selectedDataset,
      selectedAlgo,
      params,
      outputName,
    };
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
  }, [selectedDataset, selectedAlgo, params, outputName]);


  // Load datasets on component mount
  useEffect(() => {
    if (!token) return;
    setLoading(true);
    fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/datasets`, {
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${token}`
      }
    })
      .then(async res => {
        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.detail || 'Erreur lors du chargement des datasets');
        }
        const data = await res.json();
        setDatasets(data);
        if (data.length > 0) {
          setSelectedDataset(data[0].id.toString());
        }
      })
      .catch(e => setError((e as Error).message))
      .finally(() => setLoading(false));
  }, [token]);

  // Handle parameter changes
  const handleParamChange = (algo: AlgorithmKey, param: string, value: any) => {
    setParams(prev => ({
      ...prev,
      [algo]: {
        ...prev[algo],
        [param]: value,
      },
    }));
  };

  const handleRunImputation = async () => {
    // 1. Validation améliorée
    if (!selectedDataset) {
      setImputationError('Please select a dataset first.');
      return;
    }
    if (!outputName.trim()) {
      setImputationError('Please provide an output filename.');
      return;
    }

    setImputationLoading(true);
    setImputationError(null);
    setImputationResult(null);
    setTaskId(null);
    setSessionStepId(null);
    setJobStatus(null);

    // 2. Préparation du payload correct
    const originalExtension = datasets.find(ds => ds.id.toString() === selectedDataset)?.filename.split('.').pop() || 'csv';
    const fullOutputName = `${outputName.trim()}.${originalExtension}`;

    const payload = {
      dataset_id: Number(selectedDataset),
      strategy: selectedAlgo, // <-- NOM CORRIGÉ: algorithm -> strategy
      params: params[selectedAlgo],
      output_name: fullOutputName, // <-- CHAMP AJOUTÉ
      async_job: true, // now dispatch async
    };

    try {
      // 3. URL de l'endpoint corrigée + session_id
      const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const url = new URL(`${apiBase}/api/v1/imputation/run`);
      if (sessionIdFromUrl) {
        url.searchParams.set('session_id', sessionIdFromUrl);
      }
      const res = await fetch(url.toString(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) {
        const errorMessage = data.detail?.[0]?.msg || data.detail || 'Imputation failed';
        throw new Error(errorMessage);
      }

      // Expect queued response
      if (data?.status === 'queued' && data?.task_id) {
        setTaskId(data.task_id);
        setSessionStepId(data.session_step_id || null);
        setJobStatus('QUEUED');
        // Polling starts via useEffect below
      } else {
        // Fallback: if backend returned immediate result (sync)
        setImputationResult(data);
        setImputationLoading(false);
      }
    } catch (e: any) {
      setImputationError(e.message || 'Failed to start imputation');
      setImputationLoading(false);
    }
  };

  // Poll imputation task status when taskId is set
  useEffect(() => {
    if (!taskId || !token) return;

    let cancelled = false;
    const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${apiBase}/api/v1/imputation/task-status/${taskId}`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
            'Authorization': `Bearer ${token}`
          }
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Failed to fetch task status');
        }
        if (cancelled) return;

        setJobStatus(data.status || null);

        const stepStatus = data.session_step?.status as string | undefined;
        if (data.result) {
          setImputationResult(data.result);
          setImputationLoading(false);
          clearInterval(interval);
          setTaskId(null);
          return;
        }

        // Fallback: query session step status directly if available
        try {
          let stepData: any | null = null;
          if (sessionStepId) {
            const stepRes = await fetch(`${apiBase}/api/v1/sessions/steps/${sessionStepId}`, {
              method: 'GET',
              headers: { 'Accept': 'application/json', 'Authorization': `Bearer ${token}` }
            });
            if (stepRes.ok) stepData = await stepRes.json();
          } else {
            const byRefUrl = new URL(`${apiBase}/api/v1/sessions/steps/by-run-ref`);
            byRefUrl.searchParams.set('run_ref_type', 'imputation');
            byRefUrl.searchParams.set('run_ref_id', taskId);
            const stepRes = await fetch(byRefUrl.toString(), {
              method: 'GET',
              headers: { 'Accept': 'application/json', 'Authorization': `Bearer ${token}` }
            });
            if (stepRes.ok) stepData = await stepRes.json();
          }
          if (stepData) {
            const s = stepData.status as string;
            if (s === 'success') {
              setImputationLoading(false);
              clearInterval(interval);
              setTaskId(null);
              return;
            }
            if (s === 'failed') {
              setImputationError(stepData.error || 'Imputation failed');
              setImputationLoading(false);
              clearInterval(interval);
              setTaskId(null);
              return;
            }
          }
        } catch (_) {
          // ignore fallback errors
        }

        if (stepStatus === 'success') {
          // If result not provided for some reason, stop polling and let user refresh
          setImputationLoading(false);
          clearInterval(interval);
          setTaskId(null);
        } else if (stepStatus === 'failed' || data.status === 'FAILURE') {
          setImputationError(data.session_step?.error || 'Imputation failed');
          setImputationLoading(false);
          clearInterval(interval);
          setTaskId(null);
        }
      } catch (e) {
        if (cancelled) return;
        // Ignore transient errors, keep polling
      }
    }, 2000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [taskId, token, sessionStepId]);

  // Handle file download with authentication
  const handleDownloadFile = async (filePath: string) => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/datasets/file/download?path=${encodeURIComponent(filePath)}`,
        {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );

      if (!response.ok) {
        throw new Error('Failed to download file');
      }

      // Get the filename from the path
      const filename = filePath.split('\\').pop() || filePath.split('/').pop() || 'download.csv';
      
      // Create blob and download
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Download failed:', error);
      setImputationError('Failed to download file. Please try again.');
    }
  };

  // Render algorithm parameters based on selected algorithm
  const renderAlgoParams = () => {
    const currentParams = params[selectedAlgo];
    
    switch (selectedAlgo) {
      case 'simple':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Strategy</label>
              <select
                value={currentParams.strategy}
                onChange={(e) => handleParamChange('simple', 'strategy', e.target.value)}
                className="w-full p-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="mean">Mean</option>
                <option value="median">Median</option>
                <option value="most_frequent">Most Frequent</option>
              </select>
            </div>
          </div>
        );
      
      case 'mice':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Number of Imputations</label>
              <input
                type="number"
                min="1"
                max="20"
                value={currentParams.n_imputations}
                onChange={(e) => handleParamChange('mice', 'n_imputations', Number(e.target.value))}
                className="w-full p-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Max Iterations</label>
              <input
                type="number"
                min="1"
                max="100"
                value={currentParams.max_iter}
                onChange={(e) => handleParamChange('mice', 'max_iter', Number(e.target.value))}
                className="w-full p-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Random State (optional)</label>
              <input
                type="number"
                value={currentParams.random_state || ''}
                onChange={(e) => handleParamChange('mice', 'random_state', e.target.value ? Number(e.target.value) : undefined)}
                className="w-full p-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
                placeholder="42"
              />
            </div>
          </div>
        );
      
      case 'knn':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Number of Neighbors</label>
              <input
                type="number"
                min="1"
                max="20"
                value={currentParams.n_neighbors}
                onChange={(e) => handleParamChange('knn', 'n_neighbors', Number(e.target.value))}
                className="w-full p-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Weights</label>
              <select
                value={currentParams.weights}
                onChange={(e) => handleParamChange('knn', 'weights', e.target.value)}
                className="w-full p-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="uniform">Uniform</option>
                <option value="distance">Distance</option>
              </select>
            </div>
          </div>
        );
      
      case 'missforest':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Number of Estimators</label>
              <input
                type="number"
                min="10"
                max="1000"
                value={currentParams.n_estimators}
                onChange={(e) => handleParamChange('missforest', 'n_estimators', Number(e.target.value))}
                className="w-full p-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Max Depth (optional)</label>
              <input
                type="number"
                min="1"
                max="50"
                value={currentParams.max_depth || ''}
                onChange={(e) => handleParamChange('missforest', 'max_depth', e.target.value ? Number(e.target.value) : undefined)}
                className="w-full p-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
                placeholder="Auto"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Random State (optional)</label>
              <input
                type="number"
                value={currentParams.random_state || ''}
                onChange={(e) => handleParamChange('missforest', 'random_state', e.target.value ? Number(e.target.value) : undefined)}
                className="w-full p-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
                placeholder="42"
              />
            </div>
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6">Data Imputation</h1>

      {/* Dataset Selection */}
      <div className="bg-white p-6 rounded-lg shadow-md mb-6">
        <h2 className="text-xl font-semibold mb-4">Select Dataset</h2>
        {loading ? (
          <div className="text-gray-500">Loading datasets...</div>
        ) : error ? (
          <div className="text-red-500 bg-red-50 p-3 rounded-md">{error}</div>
        ) : (
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="w-full p-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
            disabled={datasets.length === 0}
          >
            <option value="" disabled>Select a dataset</option>
            {datasets.map(ds => (
              <option key={ds.id} value={ds.id.toString()}>{ds.filename}</option>
            ))}
          </select>
        )}
      </div>

      {/* Algorithm & Parameters */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold mb-4">Algorithm & Parameters</h2>
        
        {/* Algorithm Selection */}
        <div className="flex border-b mb-6">
          {algorithms.map(algo => (
            <button
              key={algo.key}
              onClick={() => setSelectedAlgo(algo.key as AlgorithmKey)}
              className={`py-2 px-4 font-medium text-sm focus:outline-none ${
                selectedAlgo === algo.key
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {algo.label}
            </button>
          ))}
        </div>

        {/* Parameters */}
        <div className="mb-6">
          {renderAlgoParams()}
        </div>

        {/* Output Filename */}
        <div className="mt-6">
          <label htmlFor="outputName" className="block text-sm font-medium text-gray-700 mb-1">
            Output Filename
          </label>
          <div className="flex items-center">
            <input
              id="outputName"
              type="text"
              value={outputName}
              onChange={(e) => setOutputName(e.target.value)}
              className="w-full p-2 border rounded-l-md focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100"
              placeholder="e.g., imputed_sales_data"
              disabled={!selectedDataset}
            />
            <span className="inline-flex items-center px-3 text-gray-500 bg-gray-100 border border-l-0 rounded-r-md h-[42px]">
              .{datasets.find(ds => ds.id.toString() === selectedDataset)?.filename.split('.').pop() || 'csv'}
            </span>
          </div>
        </div>

        {/* Run Button */}
        <div className="mt-8">
          <button
            type="button"
            className="w-full bg-blue-600 text-white font-bold py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400"
            disabled={!selectedDataset || !outputName.trim() || imputationLoading}
            onClick={handleRunImputation}
          >
            {imputationLoading ? (jobStatus ? `Imputation: ${jobStatus}...` : 'Running Imputation...') : 'Run Imputation'}
          </button>
          {taskId && (
            <div className="mt-2 text-sm text-gray-600">
              <div><span className="font-medium">Task:</span> {taskId}</div>
              {sessionStepId && <div><span className="font-medium">Session Step:</span> {sessionStepId}</div>}
            </div>
          )}
        </div>
      </div>

      {/* Missing Values Chart */}
      {imputationResult?.summary?.missing_before && (
        <div className="mt-8">
          <MissingValueChart
            missingBefore={imputationResult.summary.missing_before}
            missingAfter={imputationResult.summary.missing_after}
          />
        </div>
      )}

      {/* Results */}
      <div className="mt-4 min-h-[32px]">
        {imputationError && (
          <div className="text-red-500 bg-red-50 p-3 rounded-md">{imputationError}</div>
        )}
        {imputationResult && (
          <div className="text-green-600 bg-green-50 p-4 rounded-md">
            <div className="font-semibold">{imputationResult.message || 'Operation completed successfully!'}</div>
            {(imputationResult.imputed_values_count ?? imputationResult.summary?.total_filled) !== undefined && (
              <div><span className="font-medium">Values Imputed:</span> {(imputationResult.imputed_values_count ?? imputationResult.summary.total_filled).toLocaleString()}</div>
            )}
            {(imputationResult.imputed_rows_count ?? imputationResult.summary?.total_missing_before) !== undefined && (
              <div><span className="font-medium">Rows with Missing Before:</span> {(imputationResult.imputed_rows_count ?? imputationResult.summary.total_missing_before).toLocaleString()}</div>
            )}

            {/* Performance metrics */}
            {imputationResult.summary?.performance && (
              <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                <div><span className="font-medium">Runtime:</span> {imputationResult.summary.performance.runtime_seconds}s</div>
                {imputationResult.summary.performance.rmse !== undefined && (
                  <div><span className="font-medium">RMSE:</span> {imputationResult.summary.performance.rmse.toFixed(3)}</div>
                )}
                {imputationResult.summary.performance.mae !== undefined && (
                  <div><span className="font-medium">MAE:</span> {imputationResult.summary.performance.mae.toFixed(3)}</div>
                )}
                {imputationResult.summary.performance.cat_accuracy !== undefined && (
                  <div><span className="font-medium">Cat. Accuracy:</span> {(imputationResult.summary.performance.cat_accuracy * 100).toFixed(1)}%</div>
                )}
              </div>
            )}

            {/* Warnings */}
            {imputationResult.warnings && imputationResult.warnings.length > 0 && (
              <div className="mt-3 bg-yellow-50 border-l-4 border-yellow-400 p-2 text-yellow-800 rounded">
                <div className="font-medium mb-1">Warnings:</div>
                <ul className="list-disc list-inside space-y-1 text-sm">
                  {imputationResult.warnings.map((w: string, idx: number) => (
                    <li key={idx}>{w}</li>
                  ))}
                </ul>
              </div>
            )}
            {imputationResult.imputed_dataset_path && (
              <div className="mt-2 flex items-center gap-2">
                <span className="font-medium">New dataset created:</span> {imputationResult.imputed_dataset_path.split('\\').pop() || imputationResult.imputed_dataset_path.split('/').pop()}
                <button
                  onClick={() => handleDownloadFile(imputationResult.imputed_dataset_path)}
                  className="ml-4 bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 font-medium"
                >
                  Download
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ImputationPage;
