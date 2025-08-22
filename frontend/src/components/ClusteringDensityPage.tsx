import React, { useEffect, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useSanitizedApi } from '../hooks/useSanitizedApi';
import { useSearchParams } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// TypeScript interfaces for clustering results
interface ClusteringVisualization {
  scatter_plot_path: string;
  distribution_plot_path?: string;
  pca_plot_path?: string;
  reachability_plot_path?: string; // OPTICS specific
  density_plot_path?: string; // DENCLUE specific
}

interface ClusteringEvaluation {
  silhouette_score: number | null;
  davies_bouldin_score: number | null;
  f1_score: number | null;
  precision_score: number | null;
  recall_score: number | null;
  accuracy_score: number | null;
}

interface ClusteringResults {
  n_clusters: number;
  n_noise_points: number;
  data_shape: [number, number]; // [rows, columns]
  cluster_sizes: Record<string, number>;
}

interface ClusteringAnalysisSummary {
  algorithm: string;
  parameters: Record<string, number | string>;
  results: ClusteringResults;
  evaluation: ClusteringEvaluation;
}

interface ClusteringResultData {
  id: string;
  cluster_labels: number[];
  n_clusters: number;
  analysis_summary: ClusteringAnalysisSummary;
  visualizations: ClusteringVisualization;
}

const algoOptions = [
  { key: 'dbscan', label: 'DBSCAN' },
  { key: 'optics', label: 'OPTICS' },
  { key: 'denclue', label: 'DENCLUE' }
];

const defaultParams = {
  dbscan: { eps: 0.5, min_samples: 5 },
  optics: { min_samples: 5, xi: 0.05 },
  denclue: { bandwidth: 1.0, epsilon: 0.1 }
};

const ClusteringDensityPage: React.FC = () => {
  const { token } = useAuth();
  const { postWithSanitizedPayload } = useSanitizedApi();
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get('session_id') || localStorage.getItem('active_session_id') || '';
  const withSession = (url: string) => sessionId ? `${url}${url.includes('?') ? '&' : '?'}session_id=${encodeURIComponent(sessionId)}` : url;

  const [datasets, setDatasets] = useState<{id: number, name: string, filename: string}[]>([]);
  const [featureSets, setFeatureSets] = useState<any[]>([]);
  const [inputType, setInputType] = useState<'dataset'|'feature_set'>('dataset');
  const [selectedDataset, setSelectedDataset] = useState<number|null>(null);
  const [algorithm, setAlgorithm] = useState<'dbscan'|'optics'|'denclue'>('dbscan');
  const [params, setParams] = useState<any>(defaultParams.dbscan);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string|null>(null);
  const [results, setResults] = useState<ClusteringResultData | null>(null);
  const [downloadFilename, setDownloadFilename] = useState<string>('');
  const [uploadMode, setUploadMode] = useState<boolean>(true); // State for file upload
  const [file, setFile] = useState<File | null>(null);
  const [trueLabelFile, setTrueLabelFile] = useState<File | null>(null);

  // Fetch datasets and feature sets
  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/datasets`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (response.ok) {
          const data = await response.json();
          setDatasets(data);
        } else {
          setError('Failed to fetch datasets');
        }
      } catch (err) {
        setError('Error fetching datasets');
      }
    };
    
    const fetchFeatureSets = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/feature-sets`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (response.ok) {
          const data = await response.json();
          setFeatureSets(data);
        }
      } catch (err) {}
    };
    
    fetchDatasets();
    fetchFeatureSets();
  }, [token]);

  // Update params when algorithm changes
  useEffect(() => {
    setParams(defaultParams[algorithm]);
  }, [algorithm]);

  const handleParamChange = (key: string, value: any) => {
    setParams((prev: any) => ({ ...prev, [key]: value }));
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };
  
  const handleTrueLabelFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setTrueLabelFile(e.target.files[0]);
    }
  };

  const uploadTrueLabelFile = async (): Promise<string | null> => {
    if (!trueLabelFile) return null;
    
    try {
      // Create form data for true labels file upload
      const formData = new FormData();
      formData.append('file', trueLabelFile);
      
      // Make API call to upload true labels file
      const response = await fetch(`${API_BASE_URL}/api/v1/upload/file`, {
        method: 'POST',
        body: formData,
        credentials: 'include'
      });
      
      if (!response.ok) {
        throw new Error(`Error uploading true labels file: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.file_path; // Return the path to the uploaded file
    } catch (error) {
      console.error('Error uploading true labels file:', error);
      return null;
    }
  };

  const runClusteringWithUpload = async () => {
    if (!file) {
      setError('Please select a file first.');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResults(null);
    
    try {
      // Create form data for file upload
      const formData = new FormData();
      formData.append('algorithm', algorithm);
      formData.append('parameters', JSON.stringify(params));
      formData.append('file', file);
      
      // Add true labels file if provided
      if (trueLabelFile) {
        formData.append('true_labels_file', trueLabelFile);
      }

      // Make API call with FormData
      const response = await fetch(withSession(`${API_BASE_URL}/api/v1/clustering/density/upload`), {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to run clustering');
      }
      
      const data = await response.json();
      setResults(data);
      
      // Set default download filename based on algorithm and original filename
      const fileBaseName = file.name.split('.')[0];
      setDownloadFilename(`${algorithm}_${fileBaseName}_results.csv`);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const runClusteringWithDataset = async () => {
    if (!selectedDataset) {
      setError('Please select a dataset first.');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResults(null);
    
    try {
      const payload: any = {
        dataset_id: selectedDataset,
        algorithm,
        parameters: params
      };
      
      // Upload true labels file if provided
      if (trueLabelFile) {
        const trueLabelsPath = await uploadTrueLabelFile();
        if (trueLabelsPath) {
          payload.true_labels_path = trueLabelsPath;
        }
      }

      // Make API call
      const response = await postWithSanitizedPayload<any>(
        withSession(`${API_BASE_URL}/api/v1/clustering/density`),
        payload,
        token
      );
      setResults(response);
      
      // Set default download filename based on algorithm and dataset
      setDownloadFilename(`${algorithm}_clustering_results.csv`);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  // State to track true labels file path after upload
  const [trueLabelFilePath, setTrueLabelFilePath] = useState<string | null>(null);
  
  const runClustering = async () => {
    if (uploadMode) {
      await runClusteringWithUpload();
    } else {
      await runClusteringWithDataset();
    }
  };
  
  // Handle download of clustering results
  const handleDownload = async (clusteringId: number) => {
    if (!clusteringId) return;
    
    setLoading(true);
    try {
      // Construct the download URL with optional filename parameter
      let downloadUrl = `${API_BASE_URL}/api/v1/clustering/download/${clusteringId}`;
      const qs: string[] = [];
      if (downloadFilename && downloadFilename.trim()) {
        qs.push(`filename=${encodeURIComponent(downloadFilename.trim())}`);
      }
      if (sessionId) {
        qs.push(`session_id=${encodeURIComponent(sessionId)}`);
      }
      if (qs.length) {
        downloadUrl += `?${qs.join('&')}`;
      }
      
      // Use authenticated fetch request
      const response = await fetch(downloadUrl, {
        headers: { 'Authorization': `Bearer ${token}` },
        method: 'GET'
      });
      
      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }
      
      // Get the blob from the response
      const blob = await response.blob();
      
      // Create a download link and trigger the download
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = downloadFilename || 'clustering_results.csv';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
    } catch (err: any) {
      setError(`Download error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // UI rendering
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Clustering - Density-based methods</h1>
      {sessionId && (
        <div className="mb-2 text-xs text-gray-500">
          Active session: <span className="font-mono">{sessionId}</span>
        </div>
      )}
      {/* Mode selection */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">Data Source</h3>
        <div className="flex space-x-4">
          <button
            className={`px-4 py-2 rounded ${uploadMode ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
            onClick={() => setUploadMode(true)}
          >
            Upload New File
          </button>
          <button
            className={`px-4 py-2 rounded ${!uploadMode ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
            onClick={() => setUploadMode(false)}
          >
            Use Existing Dataset
          </button>
        </div>
      </div>

      {/* Data source selection based on mode */}
      <div className="mb-6">
        {uploadMode ? (
          <div>
            <h3 className="text-lg font-semibold mb-2">Upload Dataset</h3>
            <div className="border-2 border-dashed border-gray-300 rounded p-4">
              <input
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileChange}
                className="w-full"
              />
              {file && (
                <div className="mt-2 text-sm text-green-600">
                  Selected file: {file.name} ({(file.size / 1024).toFixed(2)} KB)
                </div>
              )}
              <p className="text-xs text-gray-500 mt-2">
                Supported formats: CSV, Excel (.xlsx, .xls)
              </p>
            </div>
            <div className="mt-4">
              <h4 className="text-lg font-medium mb-2">Ground Truth Labels (Optional)</h4>
              <input
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleTrueLabelFileChange}
                className="w-full"
              />
              {trueLabelFile && (
                <div className="mt-2 text-sm text-green-600">
                  Selected labels file: {trueLabelFile.name} ({(trueLabelFile.size / 1024).toFixed(2)} KB)
                </div>
              )}
              <p className="text-xs text-gray-500 mt-2">
                Supported formats: CSV, Excel (.xlsx, .xls)
              </p>
            </div>
          </div>
        ) : (
          <div>
            <h3 className="text-lg font-semibold mb-2">Select Dataset</h3>
            <select
              className="w-full p-2 border rounded"
              value={selectedDataset || ''}
              onChange={(e) => setSelectedDataset(e.target.value ? parseInt(e.target.value) : null)}
            >
              <option value="">-- Select a dataset --</option>
              {datasets.map(dataset => (
                <option key={dataset.id} value={dataset.id}>{dataset.filename || dataset.name}</option>
              ))}
            </select>
            
            <div className="mt-4">
              <h4 className="text-lg font-medium mb-2">Ground Truth Labels (Optional)</h4>
              <div className="border-2 border-dashed border-gray-300 rounded p-4">
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls"
                  onChange={handleTrueLabelFileChange}
                  className="w-full"
                />
                {trueLabelFile && (
                  <div className="mt-2 text-sm text-green-600">
                    Selected labels file: {trueLabelFile.name} ({(trueLabelFile.size / 1024).toFixed(2)} KB)
                  </div>
                )}
                <p className="text-xs text-gray-500 mt-2">
                  Upload ground truth labels to enable supervised evaluation metrics (F1-score, precision, recall, accuracy)
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Algorithm selection and parameters */}
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <h2 className="text-lg font-semibold mb-3">Algorithm & Parameters</h2>
        <div className="flex flex-col md:flex-row md:space-x-6">
          <div className="w-full mb-6">
            <label className="block mb-2 font-medium">Algorithm</label>
            <div className="flex space-x-2 mb-4">
              {algoOptions.map(opt => (
                <button
                  key={opt.key}
                  type="button"
                  onClick={() => setAlgorithm(opt.key as any)}
                  className={`px-4 py-2 rounded border font-semibold focus:outline-none transition-colors ${algorithm === opt.key ? 'bg-blue-600 text-white border-blue-600' : 'bg-gray-100 text-gray-700 border-gray-300 hover:bg-blue-50'}`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
            {/* Parameter inputs */}
            <div className="flex flex-col md:flex-row md:space-x-4">
              {algorithm === 'dbscan' && (
                <>
                  <div className="mb-3 md:mb-0">
                    <label className="block mb-1">eps</label>
                    <input
                      type="number"
                      step="0.01"
                      className="border rounded p-2 w-40"
                      value={params.eps}
                      onChange={e => handleParamChange('eps', parseFloat(e.target.value))}
                    />
                  </div>
                  <div>
                    <label className="block mb-1">min_samples</label>
                    <input
                      type="number"
                      className="border rounded p-2 w-40"
                      value={params.min_samples}
                      onChange={e => handleParamChange('min_samples', parseInt(e.target.value))}
                    />
                  </div>
                </>
              )}
              {algorithm === 'optics' && (
                <>
                  <div className="mb-3 md:mb-0">
                    <label className="block mb-1">min_samples</label>
                    <input
                      type="number"
                      className="border rounded p-2 w-40"
                      value={params.min_samples}
                      onChange={e => handleParamChange('min_samples', parseInt(e.target.value))}
                    />
                  </div>
                  <div>
                    <label className="block mb-1">xi</label>
                    <input
                      type="number"
                      step="0.01"
                      className="border rounded p-2 w-40"
                      value={params.xi}
                      onChange={e => handleParamChange('xi', parseFloat(e.target.value))}
                    />
                  </div>
                </>
              )}
              {algorithm === 'denclue' && (
                <>
                  <div className="mb-3 md:mb-0">
                    <label className="block mb-1">bandwidth</label>
                    <input
                      type="number"
                      step="0.01"
                      className="border rounded p-2 w-40"
                      value={params.bandwidth}
                      onChange={e => handleParamChange('bandwidth', parseFloat(e.target.value))}
                    />
                  </div>
                  <div>
                    <label className="block mb-1">epsilon</label>
                    <input
                      type="number"
                      step="0.01"
                      className="border rounded p-2 w-40"
                      value={params.epsilon}
                      onChange={e => handleParamChange('epsilon', parseFloat(e.target.value))}
                    />
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Run button */}
      <div className="mb-6">
        <button
          className="bg-blue-600 text-white px-6 py-2 rounded font-semibold shadow"
          onClick={runClustering}
          disabled={loading}
        >
          {loading ? 'Running...' : 'Run Clustering'}
        </button>
      </div>

      {/* Results */}
      {results && (
        <div className="bg-green-50 border-l-4 border-green-400 p-4 mb-4">
          <h3 className="text-xl font-semibold mb-4">Clustering Results</h3>
          
          {/* Analysis Summary */}
          {results.analysis_summary && (
            <div className="mb-6 bg-white p-4 rounded shadow-sm">
              <h4 className="text-lg font-semibold mb-3 border-b pb-2">Analysis Summary</h4>
              
              {/* Algorithm & Parameters */}
              <div className="mb-4">
                <h5 className="font-medium text-gray-700">Algorithm & Parameters</h5>
                <div className="grid grid-cols-2 gap-2 mt-2">
                  <div className="text-sm"><span className="font-medium">Algorithm:</span> {results.analysis_summary.algorithm}</div>
                  {results.analysis_summary.parameters && Object.entries(results.analysis_summary.parameters).map(([key, value]) => (
                    <div key={key} className="text-sm">
                      <span className="font-medium">{key}:</span> {String(value)}
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Results Statistics */}
              <div className="mb-4">
                <h5 className="font-medium text-gray-700">Clustering Statistics</h5>
                <div className="grid grid-cols-2 gap-2 mt-2">
                  <div className="text-sm"><span className="font-medium">Number of clusters:</span> {results.analysis_summary.results.n_clusters}</div>
                  <div className="text-sm"><span className="font-medium">Noise points:</span> {results.analysis_summary.results.n_noise_points}</div>
                  <div className="text-sm"><span className="font-medium">Data shape:</span> {results.analysis_summary.results.data_shape[0]} rows × {results.analysis_summary.results.data_shape[1]} columns</div>
                </div>
              </div>
              
              {/* Cluster Sizes */}
              {results.analysis_summary.results.cluster_sizes && (
                <div className="mb-4">
                  <h5 className="font-medium text-gray-700">Cluster Sizes</h5>
                  <div className="grid grid-cols-3 gap-2 mt-2">
                    {Object.entries(results.analysis_summary.results.cluster_sizes).map(([cluster, size]) => (
                      <div key={cluster} className="text-sm">
                        <span className="font-medium">Cluster {cluster}:</span> {String(size)} points
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Evaluation Metrics */}
              {results.analysis_summary.evaluation && (
                <div className="mb-4">
                  <h5 className="font-medium text-gray-700">Evaluation Metrics</h5>
                  
                  {/* Unsupervised Metrics */}
                  <div className="mb-2">
                    <h6 className="text-sm font-medium text-gray-600 mb-1">Unsupervised Metrics:</h6>
                    <div className="grid grid-cols-2 gap-2">
                      {results.analysis_summary.evaluation.silhouette_score !== null && (
                        <div className="text-sm">
                          <span className="font-medium">Silhouette Score:</span> {results.analysis_summary.evaluation.silhouette_score?.toFixed(4)}
                          <span className="ml-2 text-xs text-gray-500">(higher is better, range: -1 to 1)</span>
                        </div>
                      )}
                      {results.analysis_summary.evaluation.davies_bouldin_score !== null && (
                        <div className="text-sm">
                          <span className="font-medium">Davies-Bouldin Index:</span> {results.analysis_summary.evaluation.davies_bouldin_score?.toFixed(4)}
                          <span className="ml-2 text-xs text-gray-500">(lower is better)</span>
                        </div>
                      )}
                      {results.analysis_summary.evaluation.silhouette_score === null && results.analysis_summary.evaluation.davies_bouldin_score === null && (
                        <div className="text-sm text-amber-600">
                          Unsupervised metrics could not be calculated. This typically happens when there is only one cluster or when noise points are present.
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* Supervised Metrics (when available) */}
                  {(results.analysis_summary.evaluation.f1_score !== null || 
                    results.analysis_summary.evaluation.precision_score !== null || 
                    results.analysis_summary.evaluation.recall_score !== null || 
                    results.analysis_summary.evaluation.accuracy_score !== null) && (
                    <div>
                      <h6 className="text-sm font-medium text-gray-600 mb-1">Supervised Metrics (with ground truth):</h6>
                      <div className="grid grid-cols-2 gap-2">
                        {results.analysis_summary.evaluation.accuracy_score !== null && (
                          <div className="text-sm">
                            <span className="font-medium">Accuracy:</span> {results.analysis_summary.evaluation.accuracy_score?.toFixed(4)}
                          </div>
                        )}
                        {results.analysis_summary.evaluation.f1_score !== null && (
                          <div className="text-sm">
                            <span className="font-medium">F1 Score:</span> {results.analysis_summary.evaluation.f1_score?.toFixed(4)}
                          </div>
                        )}
                        {results.analysis_summary.evaluation.precision_score !== null && (
                          <div className="text-sm">
                            <span className="font-medium">Precision:</span> {results.analysis_summary.evaluation.precision_score?.toFixed(4)}
                          </div>
                        )}
                        {results.analysis_summary.evaluation.recall_score !== null && (
                          <div className="text-sm">
                            <span className="font-medium">Recall:</span> {results.analysis_summary.evaluation.recall_score?.toFixed(4)}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {/* Message when no supervised metrics are available */}
                  {results.analysis_summary.evaluation.f1_score === null && 
                   results.analysis_summary.evaluation.precision_score === null && 
                   results.analysis_summary.evaluation.recall_score === null && 
                   results.analysis_summary.evaluation.accuracy_score === null && (
                    <div className="text-sm text-gray-500 mt-2">
                      <i>Note: Supervised metrics (F1-score, precision, recall) require ground truth labels for comparison.</i>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
          
          {/* Visualizations */}
          {results.visualizations && (
            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-3 border-b pb-2">Visualizations</h4>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Scatter Plot */}
                {results.visualizations.scatter_plot_path && (
                  <div className="bg-white p-3 rounded shadow-sm">
                    <h5 className="font-medium text-gray-700 mb-2">Cluster Scatter Plot</h5>
                    <img 
                      src={withSession(`${API_BASE_URL}${results.visualizations.scatter_plot_path}`)} 
                      alt="Clusters scatter plot" 
                      className="max-w-full border rounded" 
                    />
                    <p className="text-xs text-gray-500 mt-1">Scatter plot showing clusters in the first two dimensions</p>
                  </div>
                )}
                
                {/* Distribution Plot */}
                {results.visualizations.distribution_plot_path && (
                  <div className="bg-white p-3 rounded shadow-sm">
                    <h5 className="font-medium text-gray-700 mb-2">Cluster Size Distribution</h5>
                    <img 
                      src={withSession(`${API_BASE_URL}${results.visualizations.distribution_plot_path}`)} 
                      alt="Cluster size distribution" 
                      className="max-w-full border rounded" 
                    />
                    <p className="text-xs text-gray-500 mt-1">Bar chart showing the number of points in each cluster</p>
                  </div>
                )}
                
                {/* PCA Plot */}
                {results.visualizations.pca_plot_path && (
                  <div className="bg-white p-3 rounded shadow-sm">
                    <h5 className="font-medium text-gray-700 mb-2">PCA Visualization</h5>
                    <img 
                      src={withSession(`${API_BASE_URL}${results.visualizations.pca_plot_path}`)} 
                      alt="PCA visualization" 
                      className="max-w-full border rounded" 
                    />
                    <p className="text-xs text-gray-500 mt-1">PCA-based 2D visualization of high-dimensional data</p>
                  </div>
                )}
                
                {/* Algorithm-specific plots */}
                {results.visualizations.reachability_plot_path && (
                  <div className="bg-white p-3 rounded shadow-sm">
                    <h5 className="font-medium text-gray-700 mb-2">OPTICS Reachability Plot</h5>
                    <img 
                      src={withSession(`${API_BASE_URL}${results.visualizations.reachability_plot_path}`)} 
                      alt="Reachability plot" 
                      className="max-w-full border rounded" 
                    />
                    <p className="text-xs text-gray-500 mt-1">Reachability plot showing density-based clustering structure</p>
                  </div>
                )}
                
                {results.visualizations.density_plot_path && (
                  <div className="bg-white p-3 rounded shadow-sm">
                    <h5 className="font-medium text-gray-700 mb-2">DENCLUE Density Estimation</h5>
                    <img 
                      src={withSession(`${API_BASE_URL}${results.visualizations.density_plot_path}`)} 
                      alt="Density estimation plot" 
                      className="max-w-full border rounded" 
                    />
                    <p className="text-xs text-gray-500 mt-1">Density estimation contour plot with clusters</p>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Download section */}
          <div className="mt-6 p-4 bg-white rounded shadow-sm">
            <h4 className="text-lg font-semibold mb-3">Download Results</h4>
            <div className="mb-3">
              <label htmlFor="downloadFilename" className="block mb-1 font-medium">Custom Filename (optional)</label>
              <input
                id="downloadFilename"
                type="text"
                placeholder="clustering_results.csv"
                className="w-full p-2 border rounded"
                value={downloadFilename}
                onChange={(e) => setDownloadFilename(e.target.value)}
              />
              <p className="text-xs text-gray-500 mt-1">Leave empty to use default filename</p>
            </div>
            <button
              className="bg-blue-600 text-white px-4 py-2 rounded font-medium hover:bg-blue-700 transition-colors"
              onClick={() => handleDownload(results.id)}
              disabled={loading}
            >
              Download Clustering Results
            </button>
          </div>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4 rounded">
          {error}
        </div>
      )}
    </div>
  );
};

export default ClusteringDensityPage;
