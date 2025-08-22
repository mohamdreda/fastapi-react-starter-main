import React, { useState, useEffect } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { useSanitizedApi } from '../../hooks/useSanitizedApi';
import FeatureExtractionLossCurve from './FeatureExtractionLossCurve';

// Ensure this points to your backend, e.g., 'http://localhost:8000'
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface FeatureExtractionResults {
  feature_set?: {
    id: number;
    name: string;
    description?: string;
  };
  total_samples?: number;
  latent_features_preview?: any[];
  evaluation_metrics?: {
    explained_variance_ratio?: number[];
    cumulative_explained_variance?: number[];
    reconstruction_error?: number;
    stress?: number;
    kl_divergence?: number;
    [key: string]: any;
  };
  processing_time?: number;
  outliers_detected?: number;
  [key: string]: any;
}

// Helper function to format feature set name for display
const formatFeatureSetName = (name: string): string => {
  return name
    .replace(/_/g, ' ')
    .replace(/NC(\d+)/, 'Components: $1')
    .replace(/NN(\d+)/, 'Neighbors: $1')
    .replace(/PCA/g, 'PCA -')
    .replace(/ISOMAP/g, 'ISOMAP -');
};

const FeatureExtractionPage: React.FC = () => {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const { token } = useAuth();
  const { postWithSanitizedPayload } = useSanitizedApi();
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get('session_id') || localStorage.getItem('active_session_id') || '';

  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>(datasetId || '');
  const [datasetName, setDatasetName] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<FeatureExtractionResults | null>(null);
  const [downloadFilename, setDownloadFilename] = useState('');
  const [isDownloading, setIsDownloading] = useState(false);

  const [algorithm, setAlgorithm] = useState<'pca' | 'isomap'>('pca');
  const [parameters, setParameters] = useState({
    pca_n_components: 2,
    isomap_n_components: 2,
    isomap_n_neighbors: 5
  });

  useEffect(() => {
    const fetchDatasets = async () => {
      if (!token) return;
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/datasets`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        
        if (response.ok) {
          const data = await response.json();
          console.log('DATASETS FROM API:', data);
          setDatasets(data);
          if (!selectedDataset && data.length > 0) {
            setSelectedDataset(data[0].id);
          }
        } else {
          console.error('Failed to fetch datasets');
          setError('Failed to load datasets.');
        }
      } catch (error) {
        console.error('Error fetching datasets:', error);
        setError('An error occurred while fetching datasets.');
      }
    };
    
    fetchDatasets();
  }, [token, selectedDataset]); // Added selectedDataset to dependency array

  const runFeatureExtraction = async () => {
    if (!selectedDataset) {
      setError('Please select a dataset first.');
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    setDownloadFilename('');

    try {
      // Get dataset name for the feature set name
      const selectedDatasetObj = datasets.find(d => d.id.toString() === selectedDataset);
      console.log('Selected dataset object:', selectedDatasetObj);
      
      // Safely extract dataset name from available properties
      let datasetName = 'dataset';
      if (selectedDatasetObj) {
        if (selectedDatasetObj.name) {
          datasetName = selectedDatasetObj.name.replace(/\s+/g, '_');
        } else if (selectedDatasetObj.filename) {
          datasetName = selectedDatasetObj.filename.split('.')[0].replace(/\s+/g, '_');
        } else if (selectedDatasetObj.title) {
          datasetName = selectedDatasetObj.title.replace(/\s+/g, '_');
        }
      }
      console.log('Using dataset name:', datasetName);
      
      // Format date for feature set name with timestamp to ensure uniqueness
      const now = new Date();
      const formattedDate = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`;
      const timestamp = `${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`;
      
      let endpoint = '';
      let payload: any = {};
      
      if (algorithm === 'pca') {
        endpoint = '/api/v1/feature-engineering/pca/run';
        const n_components = parameters.pca_n_components;
        
        // Create a more readable feature set name with timestamp to ensure uniqueness
        const autoFeatureSetName = `PCA_${datasetName}_NC${n_components}_${formattedDate}_${timestamp}`;
        
        payload = {
          dataset_id: parseInt(selectedDataset, 10),
          n_components,
          feature_set_name: autoFeatureSetName,
          description: `PCA with ${n_components} components extracted on ${now.toLocaleDateString()}`
        };
      } else {
        endpoint = '/api/v1/feature-engineering/isomap/run';
        const n_components = parameters.isomap_n_components;
        const n_neighbors = parameters.isomap_n_neighbors;
        
        // Create a more readable feature set name with timestamp to ensure uniqueness
        const autoFeatureSetName = `ISOMAP_${datasetName}_NC${n_components}_NN${n_neighbors}_${formattedDate}_${timestamp}`;
        
        payload = {
          dataset_id: parseInt(selectedDataset, 10),
          n_components,
          n_neighbors,
          feature_set_name: autoFeatureSetName,
          description: `ISOMAP with ${n_components} components and ${n_neighbors} neighbors extracted on ${now.toLocaleDateString()}`
        };
      }
      
      console.log("Sending POST request to:", `${API_BASE_URL}${endpoint}${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`);
      console.log("With payload:", payload);
      
      const response = await fetch(`${API_BASE_URL}${endpoint}${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        const errorMessage = errorData.detail || "Unknown error";
        
        // Check for specific error types and provide more helpful messages
        if (errorMessage.includes("already exists")) {
          throw new Error("A feature set with this configuration already exists. Please try again with different parameters.");
        } else {
          throw new Error(errorMessage);
        }
      }
      
      const data = await response.json();
      setResults(data);
      
    } catch (err: any) {
      console.error('Error running feature extraction:', err);
      setError(err.message || "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  // Function to handle downloading feature set CSV
  const handleDownloadFeatureSet = async () => {
    if (!results || !results.feature_set || !results.feature_set.id) {
      setError('No feature set available to download.');
      return;
    }
    
    try {
      setIsDownloading(true);
      const featureSetId = results.feature_set.id;
      const filename = downloadFilename ? 
        (downloadFilename.endsWith('.csv') ? downloadFilename : `${downloadFilename}.csv`) :
        `${results.feature_set.name || `${algorithm}_features`}.csv`;
        
      // Use the correct endpoint path based on the algorithm
      const downloadEndpoint = algorithm === 'pca' 
        ? `/api/v1/feature-engineering/pca/download/${featureSetId}` 
        : `/api/v1/feature-engineering/isomap/download/${featureSetId}`;
      
      console.log(`Downloading from: ${API_BASE_URL}${downloadEndpoint}?filename=${encodeURIComponent(filename)}${sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : ''}`);
        
      const res = await fetch(`${API_BASE_URL}${downloadEndpoint}?filename=${encodeURIComponent(filename)}${sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : ''}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (!res.ok) {
        // Try to get more detailed error information
        try {
          const errorData = await res.json();
          throw new Error(errorData.detail || 'Failed to download feature set.');
        } catch (jsonError) {
          throw new Error(`Download failed with status: ${res.status}`);
        }
      }
      
      const blob = await res.blob();
      if (blob.size === 0) {
        throw new Error('Downloaded file is empty. Please try again.');
      }
      
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      console.error('Download error:', err);
      setError(err.message || 'Failed to download feature set.');
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6">Feature Extraction</h1>
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
          onChange={e => setSelectedDataset(e.target.value)}
          disabled={loading}
        >
          <option value="">Select a dataset</option>
          {datasets.map((ds: any) => (
            <option key={ds.id} value={ds.id}>
              {ds.name || ds.filename || ds.id}
            </option>
          ))}
        </select>
      </div>

      {selectedDataset && (
        <div className="mb-6 bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-3">Algorithm & Parameters</h2>
          <div className="flex space-x-4 mb-6">
            <button
              onClick={() => setAlgorithm('pca')}
              className={`px-6 py-2 rounded-md ${algorithm === 'pca' ? 'bg-blue-600 text-white font-medium shadow-sm' : 'bg-gray-100 hover:bg-gray-200 text-gray-800'}`}
              disabled={loading}
            >
              PCA
            </button>
            <button
              onClick={() => setAlgorithm('isomap')}
              className={`px-6 py-2 rounded-md ${algorithm === 'isomap' ? 'bg-blue-600 text-white font-medium shadow-sm' : 'bg-gray-100 hover:bg-gray-200 text-gray-800'}`}
              disabled={loading}
            >
              ISOMAP
            </button>
          </div>

          {algorithm === 'pca' ? (
            <div className="mb-6 bg-gray-50 p-4 rounded-lg border border-gray-200">
              <h3 className="text-md font-medium mb-3">PCA Parameters</h3>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Number of Components</label>
                <input
                  type="number"
                  min="1"
                  value={parameters.pca_n_components}
                  onChange={e => {
                    const value = e.target.value;
                    if (value === '') {
                      setParameters({ ...parameters, pca_n_components: '' as any });
                    } else {
                      setParameters({ ...parameters, pca_n_components: parseInt(value, 10) || 2 });
                    }
                  }}
                  className="w-full p-2 border rounded"
                  disabled={loading}
                />
                <p className="text-xs text-gray-500 mt-1">
                  The number of principal components to extract
                </p>
              </div>
            </div>
          ) : (
            <div className="mb-6 bg-gray-50 p-4 rounded-lg border border-gray-200">
              <h3 className="text-md font-medium mb-3">ISOMAP Parameters</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Number of Components</label>
                  <input
                    type="number"
                    min="1"
                    value={parameters.isomap_n_components}
                    onChange={e => {
                      const value = e.target.value;
                      if (value === '') {
                        setParameters({ ...parameters, isomap_n_components: '' as any });
                      } else {
                        setParameters({ ...parameters, isomap_n_components: parseInt(value, 10) || 2 });
                      }
                    }}
                    className="w-full p-2 border rounded"
                    disabled={loading}
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    The number of dimensions in the embedded space
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Number of Neighbors</label>
                  <input
                    type="number"
                    min="1"
                    value={parameters.isomap_n_neighbors}
                    onChange={e => {
                      const value = e.target.value;
                      if (value === '') {
                        setParameters({ ...parameters, isomap_n_neighbors: '' as any });
                      } else {
                        setParameters({ ...parameters, isomap_n_neighbors: parseInt(value, 10) || 5 });
                      }
                    }}
                    className="w-full p-2 border rounded"
                    disabled={loading}
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Number of neighbors to consider for each point
                  </p>
                </div>
              </div>
            </div>
          )}

          <button
            onClick={runFeatureExtraction}
            disabled={loading || !selectedDataset}
            className={`w-full px-4 py-3 rounded-md text-white font-semibold transition-colors ${loading || !selectedDataset ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </div>
            ) : 'Run Feature Extraction'}
          </button>
        </div>
      )}

      {error && (
        <div className="my-4 p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm font-medium">An error occurred</p>
              <p className="text-sm mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {results && (
        <div className="mt-8 bg-white p-4 rounded-lg shadow">
          <div className="mb-4">
            <h4 className="text-lg font-medium mb-2">Feature Set Information</h4>
            <div className="bg-blue-50 p-4 rounded-md border border-blue-200">
              <p className="mb-2">
                <span className="font-semibold">Name:</span> {results.feature_set?.name ? 
                  formatFeatureSetName(results.feature_set.name)
                  : `${algorithm.toUpperCase()} Features`}
              </p>
              {results.feature_set?.description && (
                <p className="mb-2">
                  <span className="font-semibold">Description:</span> {results.feature_set.description}
                </p>
              )}
              <p className="text-sm text-blue-700">
                Feature extraction completed successfully
              </p>
            </div>
          </div>
          
          {/* Download Features Section */}
          {results.feature_set && results.feature_set.id && (
            <div className="mb-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
              <h4 className="font-medium text-lg mb-2">Download Extracted Features</h4>
              <div className="flex flex-col md:flex-row gap-3 mb-3">
                <div className="flex-grow">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Custom Filename (optional)</label>
                  <input
                    type="text"
                    value={downloadFilename}
                    onChange={(e) => setDownloadFilename(e.target.value)}
                    className="w-full p-2 border rounded"
                    placeholder={formatFeatureSetName(results.feature_set.name) || `${algorithm}_features`}
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Leave empty to use the default name: {formatFeatureSetName(results.feature_set.name) || `${algorithm}_features`}.csv
                  </p>
                </div>
                <div className="flex items-end">
                  <button
                    className="bg-green-500 text-white px-6 py-2 rounded hover:bg-green-600 flex items-center gap-2 disabled:bg-gray-400"
                    disabled={isDownloading}
                    onClick={handleDownloadFeatureSet}
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
          {results && (
            <div className="bg-white p-6 rounded-lg shadow mt-6">
              <h2 className="text-xl font-semibold mb-6">Feature Engineering Results</h2>
              
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-100 mb-6">
                <div className="font-medium mb-2">Algorithm Used</div>
                <div className="text-blue-700 font-semibold text-lg">
                  {algorithm === 'pca' ? 'Principal Component Analysis (PCA)' : 'ISOMAP Manifold Learning'}
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  {algorithm === 'pca' 
                    ? `PCA with ${parameters.pca_n_components} components` 
                    : `ISOMAP with ${parameters.isomap_n_components} components and ${parameters.isomap_n_neighbors} neighbors`}
                </p>
              </div>
              
              {/* Training Loss Curve - Similar to Autoencoder */}
              <div className="bg-white p-4 rounded-lg shadow mb-6">
                <FeatureExtractionLossCurve
                  algorithm={algorithm}
                  metrics={results.evaluation_metrics}
                />
              </div>
              
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
                      {algorithm === 'pca' ? parameters.pca_n_components : parameters.isomap_n_components}
                    </div>
                  </div>
                </div>
              </div>
              
              {/* No scatter plot visualization - using loss curve instead */}
              
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
          )}
        </div>
      )}
    </div>
  );
};

export default FeatureExtractionPage;