import React, { useState, useEffect } from 'react';
import LossCurve from './LossCurve';
import { useParams, useSearchParams } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import ResultsDisplay from '../../components/OutlierDetection/ResultsDisplay';
import { useSanitizedApi } from '../../hooks/useSanitizedApi';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface OutlierResultItem {
  is_outlier: boolean;
  original_index: number;
  outlier_score?: number;
  reconstruction_error?: number;
}

interface DetectOutliersResponse {
  message: string;
  task_id?: string;
  outlier_run_id?: number;
  run_id?: string | number;
}

interface FormattedResults {
  total_samples: number;
  outlier_count: number;
  processing_time: number;
  metrics: Record<string, number>;
  outlier_indices: number[];
  outlier_scores: number[];
  visualization_data: {
    reduced_features: boolean;
    scatter_plot_path?: string;
  };
}

const AutoencoderPage: React.FC = () => {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const { token } = useAuth();
  const { postWithSanitizedPayload } = useSanitizedApi();
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get('session_id') || localStorage.getItem('active_session_id') || '';

  const [datasets, setDatasets] = useState<any[]>([]);
  // Always store dataset ID as a string, but convert to number for API
  const [selectedDataset, setSelectedDataset] = useState<string>(datasetId || '');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any | null>(null);
  const [showSaveFeatureSet, setShowSaveFeatureSet] = useState(false);
  const [saveFeatureSetError, setSaveFeatureSetError] = useState<string | null>(null);
  const [featureSetSaved, setFeatureSetSaved] = useState<any | null>(null);
  
  // Format feature set name for display (consistent with feature extraction page)
  const formatFeatureSetName = (name: string | undefined): string => {
    if (!name) return 'autoencoder_features';
    return name
      .replace(/_/g, ' ')
      .replace(/LD(\d+)/, 'Latent Dim: $1')
      .replace(/AE_/, '')
      .replace(/\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}/, '');
  };
  const [downloadFilename, setDownloadFilename] = useState('');
  const [isDownloading, setIsDownloading] = useState(false);

  const [autoencoderConfig, setAutoencoderConfig] = useState({
    latentDim: '8',
    epochs: '25',
    batchSize: '64',
    random_state: '42'
  });

  // Fetch datasets on mount
  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/datasets`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (response.ok) {
          const data = await response.json();
          setDatasets(data);
          if (!selectedDataset && data.length > 0) {
            setSelectedDataset(String(data[0].id));
          }
        } else {
          setError('Failed to fetch datasets');
        }
      } catch (err) {
        setError('Error fetching datasets');
      }
    };
    fetchDatasets();
    // eslint-disable-next-line
  }, [token]);

  // Handler for running autoencoder
  // Unified handler for running Autoencoder using the dedicated backend endpoint
  const handleRunAutoencoder = async () => {
    setFeatureSetSaved(null);
    setShowSaveFeatureSet(false);
    setSaveFeatureSetError(null);
    setLoading(true);
    setError(null);
    setResults(null);
    setDownloadFilename('');
    const datasetIdNum = parseInt(selectedDataset, 10);
    if (isNaN(datasetIdNum)) {
      setError('Please select a valid dataset.');
      setLoading(false);
      return;
    }
    try {
      // Validate and convert inputs
      const latentDimNum = Number(autoencoderConfig.latentDim);
      const epochsNum = Number(autoencoderConfig.epochs);
      const batchSizeNum = Number(autoencoderConfig.batchSize);
      const randomStateNum = Number(autoencoderConfig.random_state);
      if (
        isNaN(latentDimNum) || isNaN(epochsNum) || isNaN(batchSizeNum) || isNaN(randomStateNum) ||
        autoencoderConfig.latentDim === '' || autoencoderConfig.epochs === '' || autoencoderConfig.batchSize === '' || autoencoderConfig.random_state === ''
      ) {
        setError('Please enter valid numbers for all Autoencoder parameters.');
        setLoading(false);
        return;
      }
      
      // Auto-generate a feature set name based on dataset and parameters
      const selectedDatasetObj = datasets.find(d => d.id.toString() === selectedDataset.toString());
      const datasetName = selectedDatasetObj ? selectedDatasetObj.filename.split('.')[0] : 'dataset';
      
      // Format the date in a more readable way with time to ensure uniqueness
      const now = new Date();
      const formattedDate = `${now.getFullYear()}${(now.getMonth() + 1).toString().padStart(2, '0')}${now.getDate().toString().padStart(2, '0')}`;
      const formattedTime = `${now.getHours().toString().padStart(2, '0')}${now.getMinutes().toString().padStart(2, '0')}${now.getSeconds().toString().padStart(2, '0')}`;
      
      // Create a more readable feature set name with timestamp to avoid duplicates
      const autoFeatureSetName = `AE_${datasetName}_LD${latentDimNum}_${formattedDate}_${formattedTime}`;
      
      const response = await fetch(`${API_BASE_URL}/api/v1/feature-engineering/autoencoder/run${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          dataset_id: datasetIdNum,
          latent_dim: latentDimNum,
          epochs: epochsNum,
          batch_size: batchSizeNum,
          random_state: randomStateNum,
          feature_set_name: autoFeatureSetName,
          description: `Autoencoder with latent_dim=${latentDimNum}, epochs=${epochsNum}`
        })
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || data || 'Failed to run Autoencoder.');
        setLoading(false);
        return;
      }
      setResults(data);
      setFeatureSetSaved(null);
      setShowSaveFeatureSet(true);
      setSaveFeatureSetError(null);
    } catch (err: any) {
      setError(err.message || 'An unknown error occurred.');
    } finally {
      setLoading(false);
    }
  };

  // Remove legacy runPipeline and polling logic


  
  const createMockResults = (): FormattedResults => {
    return {
      total_samples: 1000,
      outlier_count: 50,
      processing_time: 5.2,
      metrics: {
        precision: 0.92,
        recall: 0.85,
        f1_score: 0.88,
        auc_roc: 0.95
      },
      outlier_indices: Array.from({ length: 50 }, (_, i) => i * 20),
      outlier_scores: Array.from({ length: 50 }, () => Math.random()),
      visualization_data: {
        reduced_features: true
      }
    };
  };
  
  const handleFallbackToMockResults = () => {
    console.log('Falling back to mock results');
    setTimeout(() => {
      setResults(createMockResults());
      setError(null);
      setLoading(false);
    }, 2000);
  };

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/datasets`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        
        if (response.ok) {
          const data = await response.json();
          setDatasets(data);
          
          if (!selectedDataset && data.length > 0) {
            setSelectedDataset(data[0].id);
          }
        } else {
          console.error('Failed to fetch datasets');
        }
      } catch (error) {
        console.error('Error fetching datasets:', error);
      }
    };
    
    fetchDatasets();
  }, [token, selectedDataset]);
  
  const runPipeline = async () => {
    if (!selectedDataset) {
      setError('Please select a dataset first.');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResults(null);

    const payload = {
      dataset_id: selectedDataset,
      autoencoder_config: {
        latent_dim: autoencoderConfig.latentDim,
        epochs: autoencoderConfig.epochs,
        batch_size: autoencoderConfig.batchSize,
        random_state: autoencoderConfig.random_state
      }
    };

    try {
      const response = await postWithSanitizedPayload<DetectOutliersResponse>(
        `/api/v1/outlier-detection/autoencoder`,
        payload
      );
      
      if (response && response.task_id && response.run_id) {
        pollForResults(response.task_id, response.run_id);
      } else {
        setError('Failed to start autoencoder: No valid run ID received.');
        setLoading(false);
        if (import.meta.env.DEV) {
          handleFallbackToMockResults();
        }
      }
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'An unknown error occurred.';
      console.error('Error running autoencoder:', errorMessage);
      setError(`Error: ${errorMessage}`);
      setLoading(false);
      if (import.meta.env.DEV) {
        handleFallbackToMockResults();
      }
    }
  };
  
  const pollForResults = (taskId: string, runId: string | number) => {
    const interval = setInterval(async () => {
      try {
        interface TaskStatusResponse {
          task_id?: string;
          status: string;
          run_details?: {
            total_points_processed?: number;
            total_outliers_detected?: number;
            started_at?: string;
            completed_at?: string;
            evaluation_metrics?: any;
            outlier_results?: string | any[];
            scatter_plot_pca_path?: string;
            error_message?: string;
          };
        }
        
        const response = await fetch(`${API_BASE_URL}/api/v1/outliers/tasks/${taskId}/status`, {
          headers: { Authorization: `Bearer ${token}` }
        }).then(res => res.json());
        
        if (response && response.status === 'SUCCESS') {
          clearInterval(interval);
          
          const runDetails = response.run_details;
          if (runDetails) {
            let parsedOutlierResults: OutlierResultItem[] = [];
            if (typeof runDetails.outlier_results === 'string') {
              try {
                parsedOutlierResults = JSON.parse(runDetails.outlier_results);
              } catch (e) {
                console.error('Error parsing outlier results JSON:', e);
                setError('Failed to parse results from the server.');
                setLoading(false);
                return;
              }
            } else if (Array.isArray(runDetails.outlier_results)) {
              parsedOutlierResults = runDetails.outlier_results;
            }

            const startTime = runDetails.started_at ? new Date(runDetails.started_at).getTime() : Date.now();
            const endTime = runDetails.completed_at ? new Date(runDetails.completed_at).getTime() : Date.now();
            const processingTime = (endTime - startTime) / 1000;

            const formatted: FormattedResults = {
              total_samples: runDetails.total_points_processed || 0,
              outlier_count: runDetails.total_outliers_detected || 0,
              processing_time: processingTime,
              metrics: runDetails.evaluation_metrics || {},
              outlier_indices: parsedOutlierResults
                .filter(item => item.is_outlier)
                .map(item => item.original_index),
              outlier_scores: parsedOutlierResults
                .filter(item => item.is_outlier)
                .map(item => item.outlier_score || item.reconstruction_error || 0),
              visualization_data: {
                reduced_features: !!runDetails.scatter_plot_pca_path,
                scatter_plot_path: runDetails.scatter_plot_pca_path ? `${API_BASE_URL}/${runDetails.scatter_plot_pca_path}` : undefined,
              }
            };
            
            console.log('Setting results state with formatted data');
            setResults(formatted);
            setError(null);
          } else {
            setError('Task completed, but no run details were found.');
          }
          setLoading(false);
        } else if (response && (response.status === 'FAILURE' || response.status === 'REVOKED')) {
          clearInterval(interval);
          const errorMessage = response.run_details?.error_message || 'Task failed or was revoked.';
          setError(`Task failed: ${errorMessage}`);
          setLoading(false);
          if (import.meta.env.DEV) {
            handleFallbackToMockResults();
          }
        }
      } catch (err: any) {
        clearInterval(interval);
        const errorMessage = err.response?.data?.detail || err.message || 'An error occurred while polling for results.';
        console.error('Polling error:', errorMessage);
        setError(`Error polling for results: ${errorMessage}`);
        setLoading(false);
        if (import.meta.env.DEV) {
          handleFallbackToMockResults();
        }
      }
    }, 5000);
    
    setTimeout(() => {
      clearInterval(interval);
      if (loading) {
        setError('Polling timed out. The task is taking too long to complete.');
        setLoading(false);
        if (import.meta.env.DEV) {
          handleFallbackToMockResults();
        }
      }
    }, 300000);
  };

  const saveFeatureSet = async () => {
    if (!results) {
      setError('No results to save.');
      return;
    }

    try {
      const response = await postWithSanitizedPayload<DetectOutliersResponse>(
        `/api/v1/outlier-detection/autoencoder`,
        {
          dataset_id: selectedDataset,
          autoencoder_config: {
            latent_dim: autoencoderConfig.latentDim,
            epochs: autoencoderConfig.epochs,
            batch_size: autoencoderConfig.batchSize,
            random_state: autoencoderConfig.random_state
          },
          feature_set_name: featureSetName,
          feature_set_description: featureSetDescription,
        }
      );

      if (response && response.task_id && response.run_id) {
        setResults({ ...results, feature_set_saved: true });
      } else {
        setError('Failed to save feature set: No valid run ID received.');
      }
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'An unknown error occurred.';
      console.error('Error saving feature set:', errorMessage);
      setError(`Error: ${errorMessage}`);
    }
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6">Autoencoder Anomaly Detection</h1>
      {sessionId && (
        <div className="mb-2 text-xs text-gray-500">
          Active session: <span className="font-mono">{sessionId}</span>
        </div>
      )}
      
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Select Dataset</h2>
        <select 
          className="w-full p-2 border rounded"
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
        >
          <option value="">Select a dataset</option>
          {datasets.map(dataset => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.filename}
            </option>
          ))}
        </select>
      </div>
      
      {selectedDataset && (
        <div className="mb-6 bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-3">Autoencoder Configuration</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Latent Dimension</label>
              <input
                type="number"
                min="1"
                value={autoencoderConfig.latentDim}
                onChange={(e) => setAutoencoderConfig({...autoencoderConfig, latentDim: e.target.value})}
                onKeyDown={(e) => {
                  if (e.key === 'Backspace' || e.key === 'Delete') {
                    setAutoencoderConfig({...autoencoderConfig, latentDim: ''})
                  }
                }}
                className="w-full p-2 border rounded"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Epochs</label>
              <input
                type="number"
                min="1"
                value={autoencoderConfig.epochs}
                onChange={(e) => setAutoencoderConfig({...autoencoderConfig, epochs: e.target.value})}
                onKeyDown={(e) => {
                  if (e.key === 'Backspace' || e.key === 'Delete') {
                    setAutoencoderConfig({...autoencoderConfig, epochs: ''})
                  }
                }}
                className="w-full p-2 border rounded"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Batch Size</label>
              <input
                type="number"
                min="1"
                value={autoencoderConfig.batchSize}
                onChange={(e) => setAutoencoderConfig({...autoencoderConfig, batchSize: e.target.value})}
                onKeyDown={(e) => {
                  if (e.key === 'Backspace' || e.key === 'Delete') {
                    setAutoencoderConfig({...autoencoderConfig, batchSize: ''})
                  }
                }}
                className="w-full p-2 border rounded"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Random State</label>
              <input
                type="number"
                min="0"
                value={autoencoderConfig.random_state}
                onChange={(e) => setAutoencoderConfig({...autoencoderConfig, random_state: e.target.value})}
                onKeyDown={(e) => {
                  if (e.key === 'Backspace' || e.key === 'Delete') {
                    setAutoencoderConfig({...autoencoderConfig, random_state: ''})
                  }
                }}
                className="w-full p-2 border rounded"
              />
            </div>
          </div>
          
          {/* Feature Set Name and Description fields removed as requested */}

          <button
            onClick={handleRunAutoencoder}
            disabled={loading}
            className={`px-4 py-2 rounded text-white ${loading ? 'bg-gray-400' : 'bg-blue-500 hover:bg-blue-600'}`}
          >
            {loading ? 'Processing...' : 'Run Autoencoder'}
          </button>
        </div>
      )}
      
      {error && (
        <div className="my-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {typeof error === 'string'
            ? error
            : Array.isArray(error)
              ? error.map((e, i) => <div key={i}>{e.msg || JSON.stringify(e)}</div>)
              : JSON.stringify(error)}
        </div>
      )}
      
      {results && (
        <div className="bg-white p-6 rounded-lg shadow mt-6">
          <h2 className="text-xl font-semibold mb-4">Autoencoder Results</h2>
          
          {/* Download Features Section */}
          {results.feature_set && results.feature_set.id && (
            <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
              <h4 className="font-medium text-lg mb-2">Download Extracted Features</h4>
              <div className="flex flex-col md:flex-row gap-3 mb-3">
                <div className="flex-grow">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Custom Filename (optional)</label>
                  <input
                    type="text"
                    value={downloadFilename}
                    onChange={(e) => setDownloadFilename(e.target.value)}
                    className="w-full p-2 border rounded"
                    placeholder={formatFeatureSetName(results.feature_set.name) || 'autoencoder_features'}
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Leave empty to use the default name: {formatFeatureSetName(results.feature_set.name) || 'autoencoder_features'}.csv
                  </p>
                </div>
                <div className="flex items-end">
                  <button
                    className="bg-green-500 text-white px-6 py-2 rounded hover:bg-green-600 flex items-center gap-2 disabled:bg-gray-400"
                    disabled={isDownloading}
                    onClick={async () => {
                      try {
                        setIsDownloading(true);
                        const filename = downloadFilename ? 
                          (downloadFilename.endsWith('.csv') ? downloadFilename : `${downloadFilename}.csv`) :
                          `${formatFeatureSetName(results.feature_set.name) || 'autoencoder_features'}.csv`;
                          
                        // Use the direct autoencoder download endpoint
                        const res = await fetch(`${API_BASE_URL}/api/v1/feature-engineering/autoencoder/download/${results.feature_set.id}?filename=${encodeURIComponent(filename)}${sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : ''}`, {
                          headers: { 'Authorization': `Bearer ${token}` }
                        });
                        
                        if (!res.ok) throw new Error('Failed to download feature set.');
                        const blob = await res.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = filename;
                        document.body.appendChild(a);
                        a.click();
                        a.remove();
                        window.URL.revokeObjectURL(url);
                      } catch (err) {
                        setError('Failed to download feature set.');
                      } finally {
                        setIsDownloading(false);
                      }
                    }}
                  >
                    {isDownloading ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Downloading...
                      </>
                    ) : (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                        Download Features
                      </>
                    )}
                  </button>
                </div>
              </div>
              <p className="text-sm text-gray-600">
                These features can be used for further analysis or machine learning tasks.
              </p>
            </div>
          )}
          
          {/* Feature Engineering Results */}
          <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100 mb-6">
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-100 mb-6">
              <div className="font-medium mb-2">Algorithm Used</div>
              <div className="text-blue-700 font-semibold text-lg">
                Autoencoder Feature Extraction
              </div>
              <p className="text-sm text-gray-600 mt-2">
                Autoencoder with {autoencoderConfig.latentDim} latent dimensions, {autoencoderConfig.epochs} epochs, and {autoencoderConfig.batchSize} batch size
              </p>
            </div>
            
            {/* Training Loss Curve */}
            {results.evaluation_metrics && results.evaluation_metrics.epochs && results.evaluation_metrics.loss && (
              <div className="bg-white p-4 rounded-lg shadow mb-6">
                <h3 className="font-medium text-lg mb-4">Training Loss Curve</h3>
                <LossCurve metrics={results.evaluation_metrics} />
              </div>
            )}
            
            {/* Run Metrics */}
            <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100 mb-6">
              <h3 className="font-medium text-lg mb-4">Run Metrics</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
                  <div className="text-sm text-gray-500 mb-1">Total Samples</div>
                  <div className="text-xl font-semibold">{results.total_samples?.toLocaleString() ?? 'N/A'}</div>
                </div>
                
                <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
                  <div className="text-sm text-gray-500 mb-1">Processing Time</div>
                  <div className="text-xl font-semibold">{results.processing_time?.toFixed(2) ?? 'N/A'}<span className="text-sm font-normal ml-1">seconds</span></div>
                </div>
                
                <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
                  <div className="text-sm text-gray-500 mb-1">Feature Dimensions</div>
                  <div className="text-xl font-semibold">
                    {autoencoderConfig.latentDim}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Feature Preview Table */}
            <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100 mb-6">
              <h3 className="font-medium text-lg mb-4">Preview of Extracted Features</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      {results.latent_features_preview &&
                        Object.keys(results.latent_features_preview[0] || {}).map((col) => (
                          <th key={col} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            {col}
                          </th>
                        ))}
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {results.latent_features_preview &&
                      results.latent_features_preview.map((row: any, idx: number) => (
                        <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                          {Object.values(row).map((val: any, i: number) => (
                            <td key={i} className="px-4 py-3 whitespace-nowrap text-sm text-gray-700">
                              {typeof val === 'number' ? val.toFixed(4) : val}
                            </td>
                          ))}
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-gray-500 mt-4">
                Showing {results.latent_features_preview?.length || 0} of {results.total_samples || 'unknown'} total samples
              </p>
            </div>
          </div>
        </div>
      )}

      

    </div>
  );
};

export default AutoencoderPage;