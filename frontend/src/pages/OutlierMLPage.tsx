import React, { useEffect, useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useSearchParams } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Utility function to handle image paths or base64 data
const getImageSrc = (path: string | null): string | null => {
  if (!path) return null;
  
  // Check if it's a base64 image
  if (path.startsWith('data:image/') || path.includes('base64')) {
    return path; // It's already a data URL
  }
  
  // Check if it looks like a base64 string without the data:image prefix
  if (path.length > 100 && /^[A-Za-z0-9+/=]+$/.test(path)) {
    return `data:image/png;base64,${path}`;
  }
  
  // Regular path - add API_BASE_URL if it's a relative path
  return path.startsWith('/') ? `${API_BASE_URL}${path}` : path;
};

const ML_ALGORITHMS = [
  { key: 'isolation_forest', label: 'Isolation Forest' },
  { key: 'local_outlier_factor', label: 'Local Outlier Factor (LOF)' },
  { key: 'one_class_svm', label: 'One-Class SVM' },
];

const OutlierMLPage: React.FC = () => {
  const { token } = useAuth();
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get('session_id') || localStorage.getItem('active_session_id');
  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [selectedAlgo, setSelectedAlgo] = useState('isolation_forest');
  const [params, setParams] = useState<Record<string, any>>({
    isolation_forest: { contamination: 0.01, n_estimators: 200, max_samples: 256 },
    local_outlier_factor: { n_neighbors: 20, contamination: 0.01, algorithm: 'auto', leaf_size: 30 },
    one_class_svm: { nu: 0.1, kernel: 'rbf', gamma: 'scale' },
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  
  // New state variables for file upload functionality
  const [uploadMode, setUploadMode] = useState<boolean>(false);
  const [file, setFile] = useState<File | null>(null);
  const [trueLabelFile, setTrueLabelFile] = useState<File | null>(null);
  const [downloadFilename, setDownloadFilename] = useState<string>('');

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/datasets`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (response.ok) {
          const data = await response.json();
          setDatasets(data);
          if (!selectedDataset && data.length > 0 && !uploadMode) setSelectedDataset(data[0].id);
        }
      } catch (e) { /* ignore */ }
    };
    fetchDatasets();
    // eslint-disable-next-line
  }, [token, uploadMode]);

  const handleParamChange = (algo: string, param: string, value: any) => {
    setParams((prev) => ({ ...prev, [algo]: { ...prev[algo], [param]: value } }));
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

  const runDetection = async () => {
    if (uploadMode) {
      if (!file) return setError('Please select a dataset file');
      await runDetectionWithUpload();
    } else {
      if (!selectedDataset) return setError('Please select a dataset');
      await runDetectionWithExistingDataset();
    }
  };
  
  const runDetectionWithUpload = async () => {
    setLoading(true); setError(null); setResults(null);
    try {
      // Create form data for file upload
      const formData = new FormData();
      formData.append('algorithm', selectedAlgo);
      formData.append('parameters', JSON.stringify(params[selectedAlgo]));
      formData.append('file', file!);
      formData.append('save_visualizations', 'true');
      formData.append('include_visualizations', 'true');
      
      // Add true labels file if provided
      if (trueLabelFile) {
        formData.append('true_labels_file', trueLabelFile);
      }

      // Make API call with FormData (append session_id if present)
      const uploadUrl = new URL(`${API_BASE_URL}/api/v1/outliers/upload/detect`);
      if (sessionId) uploadUrl.searchParams.set('session_id', sessionId);
      const response = await fetch(uploadUrl.toString(), {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to run outlier detection');
      }
      
      const data = await response.json();
      
      // Set default download filename based on algorithm and original filename
      if (file) {
        const fileBaseName = file.name.split('.')[0];
        setDownloadFilename(`${selectedAlgo}_${fileBaseName}_outliers.csv`);
      }
      
      // Poll for task completion if we have a run ID
      if (data.run_id) {
        // Use the same checkStatus function that's defined below
        const checkTaskStatus = async (runId: string) => {
          const statusResponse = await fetch(`${API_BASE_URL}/api/v1/outliers/runs/${runId}/status`, {
            headers: { 'Authorization': `Bearer ${token}` }
          });
          
          if (!statusResponse.ok) {
            throw new Error(`Status check failed: ${statusResponse.status}`);
          }
          
          const statusData = await statusResponse.json();
          console.log('Status data received:', statusData);
          
          // Add detailed debugging for visualization paths
          console.log('Visualization paths in response:', {
            'statusData.scatter_plot_pca_path': statusData.scatter_plot_pca_path,
            'statusData.pca_plot_path': statusData.pca_plot_path,
            'statusData.scatter_plot_path': statusData.scatter_plot_path,
            'statusData.outlier_distribution_path': statusData.outlier_distribution_path,
            'run_details paths': statusData.run_details ? {
              'scatter_plot_pca_path': statusData.run_details.scatter_plot_pca_path,
              'pca_plot_path': statusData.run_details.pca_plot_path,
              'scatter_plot_path': statusData.run_details.scatter_plot_path,
              'outlier_distribution_path': statusData.run_details.outlier_distribution_path
            } : 'No run_details'
          });
          
          // Add detailed debugging for evaluation metrics
          console.log('Evaluation metrics in response:', {
            'statusData.evaluation_metrics': statusData.evaluation_metrics,
            'run_details metrics': statusData.run_details ? {
              'evaluation_metrics': statusData.run_details.evaluation_metrics,
              'evaluation_metrics_json': statusData.run_details.evaluation_metrics_json,
              'evaluation_metrics_json type': statusData.run_details.evaluation_metrics_json ? typeof statusData.run_details.evaluation_metrics_json : 'undefined',
              'evaluation_metrics_json value': statusData.run_details.evaluation_metrics_json
            } : 'No run_details'
          });
          
          if (statusData.status === 'completed') {
            // The backend doesn't return a 'result' field, it includes the results directly
            // or nested under run_details
            
            // Check if we have run_details and extract data from there if needed
            const runDetails = statusData.run_details || {};
            
            // Extract the results from the response, checking both top-level and nested fields
            const formattedResults = {
              // Map to the fields expected by the rendering code
              total_points: statusData.total_points_processed || runDetails.total_points_processed || 0,
              num_outliers: statusData.total_outliers_detected || runDetails.total_outliers_detected || 0,
              algorithm: selectedAlgo,
              parameters: params[selectedAlgo],
              // Include the visualization paths - check both top-level and nested in run_details
              pca_plot_path: (() => {
                // Try all possible paths in order of preference
                const possiblePaths = [
                  statusData.pca_plot_path,
                  statusData.scatter_plot_pca_path,
                  runDetails?.pca_plot_path,
                  runDetails?.scatter_plot_pca_path,
                  statusData.scatter_plot_path,
                  runDetails?.scatter_plot_path
                ];
                
                // Find the first non-empty path
                const path = possiblePaths.find(p => p && typeof p === 'string');
                
                // Log the path for debugging
                if (path) {
                  console.log('Selected PCA plot path:', path.substring(0, 50) + '...');
                  
                  // Check if it's a base64 image
                  if (path.length > 100 && /^[A-Za-z0-9+/=]+$/.test(path)) {
                    console.log('Detected base64 image data for PCA plot');
                  }
                }
                
                return path || null;
              })(),
              
              // Also add scatter_plot_path separately since the UI checks for both
              scatter_plot_path: (() => {
                const path = statusData.scatter_plot_path || runDetails?.scatter_plot_path || null;
                
                // Log the path for debugging
                if (path) {
                  console.log('Selected scatter plot path:', path.substring(0, 50) + '...');
                  
                  // Check if it's a base64 image
                  if (path.length > 100 && /^[A-Za-z0-9+/=]+$/.test(path)) {
                    console.log('Detected base64 image data for scatter plot');
                  }
                }
                
                return path;
              })(),
              
              outlier_distribution_path: (() => {
                const path = statusData.outlier_distribution_path || runDetails?.outlier_distribution_path || null;
                
                // Log the path for debugging
                if (path) {
                  console.log('Selected outlier distribution path:', path.substring(0, 50) + '...');
                  
                  // Check if it's a base64 image
                  if (path.length > 100 && /^[A-Za-z0-9+/=]+$/.test(path)) {
                    console.log('Detected base64 image data for outlier distribution');
                  }
                }
                
                return path;
              })(),
              // Include evaluation metrics - check both top-level and nested
              evaluation_metrics: (() => {
                // First try to get from top-level
                if (statusData.evaluation_metrics) {
                  console.log('Using top-level evaluation_metrics:', JSON.stringify(statusData.evaluation_metrics));
                  // Add source information if it's not already present
                  let metrics = statusData.evaluation_metrics;
                  if (typeof metrics === 'string') {
                    try {
                      metrics = JSON.parse(metrics);
                    } catch (e) {
                      console.error('Error parsing evaluation_metrics string:', e);
                    }
                  }
                  
                  // Add source information for One-Class SVM metrics from original dataset
                  if (selectedAlgo === 'one_class_svm' && !metrics.source) {
                    metrics.source = 'original_dataset';
                  }
                  return metrics;
                }
                
                // Then try to get from runDetails
                if (runDetails && runDetails.evaluation_metrics) {
                  console.log('Using runDetails.evaluation_metrics:', JSON.stringify(runDetails.evaluation_metrics));
                  // Add source information if it's not already present
                  let metrics = runDetails.evaluation_metrics;
                  if (typeof metrics === 'string') {
                    try {
                      metrics = JSON.parse(metrics);
                    } catch (e) {
                      console.error('Error parsing runDetails.evaluation_metrics string:', e);
                    }
                  }
                  
                  // Add source information for One-Class SVM metrics from original dataset
                  if (selectedAlgo === 'one_class_svm' && !metrics.source) {
                    metrics.source = 'original_dataset';
                  }
                  return metrics;
                }
                
                // Then try to parse from evaluation_metrics_json
                if (runDetails && runDetails.evaluation_metrics_json) {
                  console.log('Using runDetails.evaluation_metrics_json');
                  try {
                    let metrics;
                    if (typeof runDetails.evaluation_metrics_json === 'string') {
                      metrics = JSON.parse(runDetails.evaluation_metrics_json);
                      console.log('Successfully parsed evaluation_metrics_json:', metrics);
                    } else {
                      console.log('evaluation_metrics_json is already an object:', runDetails.evaluation_metrics_json);
                      metrics = runDetails.evaluation_metrics_json;
                    }
                    
                    // Add source information for One-Class SVM metrics from original dataset
                    if (selectedAlgo === 'one_class_svm' && !metrics.source) {
                      metrics.source = 'original_dataset';
                    }
                    return metrics;
                  } catch (e) {
                    console.error('Error parsing evaluation_metrics_json:', e);
                  }
                }
                
                // Try to extract metrics from outlier_results if they exist there
                if (statusData.outlier_results || (runDetails && runDetails.outlier_results)) {
                  const outlierResults = statusData.outlier_results || runDetails?.outlier_results;
                  console.log('Trying to extract metrics from outlier_results');
                  
                  try {
                    let results = outlierResults;
                    if (typeof outlierResults === 'string') {
                      results = JSON.parse(outlierResults);
                    }
                    
                    if (results && results.evaluation_metrics) {
                      console.log('Found evaluation_metrics in outlier_results:', results.evaluation_metrics);
                      let metrics = results.evaluation_metrics;
                      if (typeof metrics === 'string') {
                        try {
                          metrics = JSON.parse(metrics);
                        } catch (e) {
                          console.error('Error parsing metrics from outlier_results:', e);
                        }
                      }
                      
                      // Add source information for One-Class SVM metrics from original dataset
                      if (selectedAlgo === 'one_class_svm' && metrics && !metrics.source) {
                        metrics.source = 'original_dataset';
                      }
                      return metrics;
                    }
                  } catch (e) {
                    console.error('Error extracting metrics from outlier_results:', e);
                  }
                }
                
                // Try to extract from parameters_json if available
                if (statusData.parameters_json || (runDetails && runDetails.parameters_json)) {
                  const params = statusData.parameters_json || runDetails?.parameters_json;
                  console.log('Trying to extract metrics from parameters_json');
                  
                  try {
                    let paramsObj = params;
                    if (typeof params === 'string') {
                      paramsObj = JSON.parse(params);
                    }
                    
                    if (paramsObj && paramsObj.evaluation_metrics) {
                      console.log('Found evaluation_metrics in parameters_json:', paramsObj.evaluation_metrics);
                      return paramsObj.evaluation_metrics;
                    }
                  } catch (e) {
                    console.error('Error extracting metrics from parameters_json:', e);
                  }
                }
                
                // If we have ground truth data, we should have evaluation metrics
                // Check if there's a true_labels field in the response
                if (statusData.true_labels || (runDetails && runDetails.true_labels)) {
                  console.log('Found true_labels but no evaluation metrics');
                  // This is a case where we should have metrics but don't
                  // Return placeholder metrics with a note
                  return {
                    precision: 0,
                    recall: 0,
                    f1_score: 0,
                    accuracy: 0,
                    note: 'Metrics calculation pending. Upload ground truth data to evaluate.'  
                  };
                }
                
                // Default metrics if nothing is found
                console.log('Using default evaluation metrics');
                return {
                  precision: 0,
                  recall: 0,
                  f1_score: 0,
                  accuracy: 0
                };
              })(),
              // Include the raw data for potential use
              outlier_results: statusData.outlier_results ||
                              (runDetails ? runDetails.outlier_results : null) ||
                              (runDetails && runDetails.outlier_results_json ?
                                (typeof runDetails.outlier_results_json === 'string' ?
                                  JSON.parse(runDetails.outlier_results_json) : runDetails.outlier_results_json) : []),
            };
            
            // Add more detailed debugging of the final formatted results
            console.log('Formatted results:', formattedResults);
            console.log('Visualization paths used:', {
              pca_plot_path: formattedResults.pca_plot_path,
              scatter_plot_path: formattedResults.scatter_plot_path,
              outlier_distribution_path: formattedResults.outlier_distribution_path
            });
            console.log('Evaluation metrics used:', formattedResults.evaluation_metrics);
            
            setResults(formattedResults);
            setLoading(false);
          } else if (statusData.status === 'failed') {
            setError(statusData.error || 'Task failed');
            setLoading(false);
          } else {
            // Still processing, check again in 2 seconds
            setTimeout(() => checkTaskStatus(runId), 2000);
          }
        };
        
        checkTaskStatus(data.run_id);
      } else {
        setError('No run ID returned from API');
        setLoading(false);
      }
    } catch (e: any) {
      setError(`Failed to run detection: ${e.message}`);
      setLoading(false);
    }
  };

  const runDetectionWithExistingDataset = async () => {
    setLoading(true); setError(null); setResults(null);
    try {
      // Prepare request payload based on selected algorithm
      const requestPayload: any = {
        algorithm: selectedAlgo,
        parameters: params[selectedAlgo],
        save_visualizations: true,
        include_visualizations: true
      };
      
      // Call the outlier detection API endpoint (append session_id if present)
      const detectUrl = new URL(`${API_BASE_URL}/api/v1/outliers/datasets/${selectedDataset}/detect`);
      if (sessionId) detectUrl.searchParams.set('session_id', sessionId);
      const response = await fetch(detectUrl.toString(), {
        method: 'POST',
        headers: { 
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestPayload)
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Poll for task completion
      const checkStatus = async (runId: string) => {
        const statusResponse = await fetch(`${API_BASE_URL}/api/v1/outliers/runs/${runId}/status`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        
        if (!statusResponse.ok) {
          throw new Error(`Status check failed: ${statusResponse.status}`);
        }
        
        const statusData = await statusResponse.json();
        
        if (statusData.status === 'completed') {
          // Build results in the same way as upload flow (backend doesn't return a 'result' field)
          const runDetails = statusData.run_details || {};
          
          const formattedResults = {
            total_points: statusData.total_points_processed || runDetails.total_points_processed || 0,
            num_outliers: statusData.total_outliers_detected || runDetails.total_outliers_detected || 0,
            algorithm: selectedAlgo,
            parameters: params[selectedAlgo],
            pca_plot_path: (() => {
              const possiblePaths = [
                statusData.pca_plot_path,
                statusData.scatter_plot_pca_path,
                runDetails?.pca_plot_path,
                runDetails?.scatter_plot_pca_path,
                statusData.scatter_plot_path,
                runDetails?.scatter_plot_path
              ];
              const path = possiblePaths.find((p: any) => p && typeof p === 'string');
              return path || null;
            })(),
            scatter_plot_path: statusData.scatter_plot_path || runDetails?.scatter_plot_path || null,
            outlier_distribution_path: statusData.outlier_distribution_path || runDetails?.outlier_distribution_path || null,
            evaluation_metrics: (() => {
              // Prefer top-level if available
              if (statusData.evaluation_metrics) {
                let metrics = statusData.evaluation_metrics;
                if (typeof metrics === 'string') {
                  try { metrics = JSON.parse(metrics); } catch (e) { /* noop */ }
                }
                if (selectedAlgo === 'one_class_svm' && metrics && !metrics.source) {
                  metrics.source = 'original_dataset';
                }
                return metrics;
              }
              // Fallback to run_details.evaluation_metrics
              if (runDetails && runDetails.evaluation_metrics) {
                let metrics = runDetails.evaluation_metrics;
                if (typeof metrics === 'string') {
                  try { metrics = JSON.parse(metrics); } catch (e) { /* noop */ }
                }
                if (selectedAlgo === 'one_class_svm' && metrics && !metrics.source) {
                  metrics.source = 'original_dataset';
                }
                return metrics;
              }
              // Fallback to evaluation_metrics_json
              if (runDetails && runDetails.evaluation_metrics_json) {
                try {
                  let metrics = runDetails.evaluation_metrics_json;
                  if (typeof metrics === 'string') metrics = JSON.parse(metrics);
                  if (selectedAlgo === 'one_class_svm' && metrics && !metrics.source) {
                    metrics.source = 'original_dataset';
                  }
                  return metrics;
                } catch (e) { /* noop */ }
              }
              // Try to extract from outlier_results
              if (statusData.outlier_results || (runDetails && runDetails.outlier_results)) {
                try {
                  let results = statusData.outlier_results || runDetails.outlier_results;
                  if (typeof results === 'string') results = JSON.parse(results);
                  if (results && results.evaluation_metrics) {
                    let metrics = results.evaluation_metrics;
                    if (typeof metrics === 'string') {
                      try { metrics = JSON.parse(metrics); } catch (e) { /* noop */ }
                    }
                    if (selectedAlgo === 'one_class_svm' && metrics && !metrics.source) {
                      metrics.source = 'original_dataset';
                    }
                    return metrics;
                  }
                } catch (e) { /* noop */ }
              }
              // Default empty metrics
              return { precision: 0, recall: 0, f1_score: 0, accuracy: 0 };
            })(),
            outlier_results: statusData.outlier_results ||
                            (runDetails ? runDetails.outlier_results : null) ||
                            (runDetails && runDetails.outlier_results_json ?
                              (typeof runDetails.outlier_results_json === 'string' ?
                                JSON.parse(runDetails.outlier_results_json) : runDetails.outlier_results_json) : []),
            run_id: runId
          };
          
          setResults(formattedResults);
          setLoading(false);
        } else if (statusData.status === 'failed') {
          setError(statusData.error || 'Task failed');
          setLoading(false);
        } else {
          // Still processing, check again in 2 seconds
          setTimeout(() => checkStatus(runId), 2000);
        }
      };
      
      // Start polling if we have a run ID
      if (data.run_id) {
        checkStatus(data.run_id);
      } else {
        setError('No run ID returned from API');
        setLoading(false);
      }
    } catch (e: any) {
      setError(`Failed to run detection: ${e.message}`);
      setLoading(false);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">ML-based Outlier Detection</h1>
      
      {/* Data Source Selection */}
      <div className="mb-6">
        <h2 className="text-lg font-semibold mb-3">Data Source</h2>
        <div className="flex gap-4 mb-4">
          <button
            onClick={() => setUploadMode(false)}
            className={`px-4 py-2 rounded font-semibold border ${!uploadMode ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-700'}`}
          >
            Use Existing Dataset
          </button>
          <button
            onClick={() => setUploadMode(true)}
            className={`px-4 py-2 rounded font-semibold border ${uploadMode ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-700'}`}
          >
            Upload New File
          </button>
        </div>
      </div>
      
      {/* Dataset Selection or Upload */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        {uploadMode ? (
          <div>
            <h2 className="text-lg font-semibold mb-3">Upload Dataset</h2>
            <div className="mb-4">
              <input
                type="file"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-500
                  file:mr-4 file:py-2 file:px-4 file:rounded
                  file:border-0 file:text-sm file:font-semibold
                  file:bg-blue-50 file:text-blue-700
                  hover:file:bg-blue-100"
              />
              <p className="mt-1 text-xs text-gray-500">Supported formats: CSV, Excel (.xls, .xlsx)</p>
            </div>
            
            <h3 className="font-medium mb-2">Ground Truth Labels (Optional)</h3>
            <div>
              <input
                type="file"
                onChange={handleTrueLabelFileChange}
                className="block w-full text-sm text-gray-500
                  file:mr-4 file:py-2 file:px-4 file:rounded
                  file:border-0 file:text-sm file:font-semibold
                  file:bg-blue-50 file:text-blue-700
                  hover:file:bg-blue-100"
              />
              <p className="mt-1 text-xs text-gray-500">Supported formats: CSV, Excel (.xls, .xlsx)</p>
            </div>
          </div>
        ) : (
          <div>
            <h2 className="text-lg font-semibold mb-3">Select Dataset</h2>
            <select
              className="w-full p-2 border rounded"
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              disabled={uploadMode}
            >
              <option value="">Select a dataset</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>{d.filename}</option>
              ))}
            </select>
          </div>
        )}
      </div>
      {/* Sélection de l'algorithme */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Select Algorithm</h2>
        <div className="flex gap-4 mb-4">
          {ML_ALGORITHMS.map((algo) => (
            <button
              key={algo.key}
              onClick={() => setSelectedAlgo(algo.key)}
              className={`px-4 py-2 rounded font-semibold border ${selectedAlgo === algo.key ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-700'}`}
            >
              {algo.label}
            </button>
          ))}
        </div>
        {/* Paramètres dynamiques */}
        {selectedAlgo === 'isolation_forest' && (
          <div className="space-y-2">
            <label className="block">Contamination
              <input type="number" step="0.01" min="0" max="1" className="ml-2 border rounded p-1 w-24" value={params.isolation_forest.contamination} onChange={e => handleParamChange('isolation_forest', 'contamination', parseFloat(e.target.value))} />
            </label>
            <label className="block">n_estimators
              <input type="number" className="ml-2 border rounded p-1 w-24" value={params.isolation_forest.n_estimators} onChange={e => handleParamChange('isolation_forest', 'n_estimators', parseInt(e.target.value))} />
            </label>
            <label className="block">max_samples
              <input type="number" className="ml-2 border rounded p-1 w-24" value={params.isolation_forest.max_samples} onChange={e => handleParamChange('isolation_forest', 'max_samples', parseInt(e.target.value))} />
            </label>
          </div>
        )}
        {selectedAlgo === 'local_outlier_factor' && (
          <div className="space-y-2">
            <label className="block">n_neighbors
              <input type="number" min="1" className="ml-2 border rounded p-1 w-24" value={params.local_outlier_factor.n_neighbors} onChange={e => handleParamChange('local_outlier_factor', 'n_neighbors', parseInt(e.target.value))} />
            </label>
            <label className="block">Contamination
              <input type="number" step="0.01" min="0" max="1" className="ml-2 border rounded p-1 w-24" value={params.local_outlier_factor.contamination} onChange={e => handleParamChange('local_outlier_factor', 'contamination', parseFloat(e.target.value))} />
            </label>
            <label className="block">Algorithm
              <select className="ml-2 border rounded p-1 w-32" value={params.local_outlier_factor.algorithm} onChange={e => handleParamChange('local_outlier_factor', 'algorithm', e.target.value)}>
                <option value="auto">auto</option>
                <option value="ball_tree">ball_tree</option>
                <option value="kd_tree">kd_tree</option>
                <option value="brute">brute</option>
              </select>
            </label>
            <label className="block">leaf_size
              <input type="number" min="1" className="ml-2 border rounded p-1 w-24" value={params.local_outlier_factor.leaf_size} onChange={e => handleParamChange('local_outlier_factor', 'leaf_size', parseInt(e.target.value))} />
            </label>
          </div>
        )}
        {selectedAlgo === 'one_class_svm' && (
          <div className="space-y-2">
            <label className="block">nu
              <input type="number" step="0.01" min="0" max="1" className="ml-2 border rounded p-1 w-24" value={params.one_class_svm.nu} onChange={e => handleParamChange('one_class_svm', 'nu', parseFloat(e.target.value))} />
            </label>
            <label className="block">kernel
              <select className="ml-2 border rounded p-1 w-32" value={params.one_class_svm.kernel} onChange={e => handleParamChange('one_class_svm', 'kernel', e.target.value)}>
                <option value="rbf">rbf</option>
                <option value="linear">linear</option>
                <option value="poly">poly</option>
                <option value="sigmoid">sigmoid</option>
              </select>
            </label>
            <label className="block">gamma
              <select className="ml-2 border rounded p-1 w-24" value={params.one_class_svm.gamma} onChange={e => handleParamChange('one_class_svm', 'gamma', e.target.value)}>
                <option value="scale">scale</option>
                <option value="auto">auto</option>
              </select>
            </label>
          </div>
        )}
      </div>
      <div className="flex justify-between items-center mt-6">
        <button
          onClick={runDetection}
          disabled={loading}
          className="px-6 py-2 bg-blue-500 text-white rounded font-semibold disabled:bg-gray-400"
        >
          {loading ? 'Processing...' : 'Run Outlier Detection'}
        </button>
        
        {/* Download button that uses downloadFilename */}
        {results && (
          <button
            onClick={() => {
              if (results.run_id) {
                window.open(`${API_BASE_URL}/api/v1/outliers/runs/${results.run_id}/download`, '_blank');
              }
            }}
            className="px-4 py-2 bg-green-500 text-white rounded font-semibold"
            title={`Download results as ${downloadFilename}`}
          >
            Download Results CSV
          </button>
        )}
      </div>
      {/* Affichage erreur/resultat */}
      {error && <div className="my-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">{error}</div>}
      {results && (
        <div className="mt-6 bg-white p-4 rounded shadow">
          <h3 className="text-xl font-semibold mb-4">Outlier Detection Results</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <h4 className="font-semibold">Summary</h4>
              <div>Detected outliers: <span className="font-bold">{results.num_outliers || 0}</span></div>
              <div>Total data points: <span className="font-bold">{results.total_points || 0}</span></div>
              <div>Outlier percentage: <span className="font-bold">
                {results.num_outliers && results.total_points ? 
                  ((results.num_outliers / results.total_points) * 100).toFixed(2) + '%' : 
                  'N/A'}
              </span></div>
            </div>
            
            <div className="space-y-2">
              <h4 className="font-semibold">Algorithm</h4>
              <div>Method: <span className="font-bold">{results.algorithm || selectedAlgo}</span></div>
              <div>Parameters: <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto">
                {JSON.stringify(results.parameters || params[selectedAlgo], null, 2)}
              </pre></div>
            </div>
          </div>
          
          {/* Visualizations */}
          <div className="mt-6">
            <h4 className="font-semibold text-lg mb-3">Visualizations</h4>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
              {/* PCA Plot */}
              {results.pca_plot_path && (
                <div className="border rounded-lg p-3 bg-gray-50">
                  <h5 className="text-md font-medium mb-2">PCA Visualization</h5>
                  {results.pca_plot_path ? (
                    <>
                      <img 
                        src={getImageSrc(results.pca_plot_path) || ''} 
                        alt="PCA Visualization" 
                        className="w-full border rounded shadow-sm"
                        onError={(e) => {
                          console.error('Error loading PCA plot image');
                          e.currentTarget.style.display = 'none';
                          const errorDiv = document.createElement('div');
                          errorDiv.className = 'text-sm text-red-500 p-3 border border-red-300 rounded bg-red-50 my-2';
                          errorDiv.textContent = 'Unable to display visualization. The image data may be in an unsupported format.';
                          e.currentTarget.parentNode?.insertBefore(errorDiv, e.currentTarget.nextSibling);
                        }}
                      />
                      {/* Only show path in development mode */}
                      {import.meta.env.DEV && (
                        <div className="text-xs text-gray-500 mt-1">Path: {results.pca_plot_path?.substring(0, 50)}...</div>
                      )}
                    </>
                  ) : (
                    <div className="text-sm text-red-500">PCA visualization not available</div>
                  )}
                  <p className="text-xs text-gray-500 mt-2">PCA reduces dimensions to visualize clusters and outliers in 2D space</p>
                </div>
              )}
              
              {/* Outlier Score Distribution */}
              {results.outlier_distribution_path && (
                <div className="border rounded-lg p-3 bg-gray-50">
                  <h5 className="text-md font-medium mb-2">Outlier Score Distribution</h5>
                  {results.outlier_distribution_path ? (
                    <>
                      <img 
                        src={getImageSrc(results.outlier_distribution_path) || ''} 
                        alt="Outlier Score Distribution" 
                        className="w-full border rounded shadow-sm"
                        onError={(e) => {
                          console.error('Error loading outlier distribution image');
                          e.currentTarget.style.display = 'none';
                          const errorDiv = document.createElement('div');
                          errorDiv.className = 'text-sm text-red-500 p-3 border border-red-300 rounded bg-red-50 my-2';
                          errorDiv.textContent = 'Unable to display visualization. The image data may be in an unsupported format.';
                          e.currentTarget.parentNode?.insertBefore(errorDiv, e.currentTarget.nextSibling);
                        }}
                      />
                      {/* Only show path in development mode */}
                      {import.meta.env.DEV && (
                        <div className="text-xs text-gray-500 mt-1">Path: {results.outlier_distribution_path?.substring(0, 50)}...</div>
                      )}
                    </>
                  ) : (
                    <div className="text-sm text-red-500">Outlier distribution visualization not available</div>
                  )}
                  <p className="text-xs text-gray-500 mt-2">Distribution of anomaly scores across all data points</p>
                </div>
              )}
              
              {/* Scatter Plot - will be available if the backend provides it */}
              {results.scatter_plot_path && (
                <div className="border rounded-lg p-3 bg-gray-50">
                  <h5 className="text-md font-medium mb-2">Scatter Plot Visualization</h5>
                  {results.scatter_plot_path ? (
                    <>
                      <img 
                        src={getImageSrc(results.scatter_plot_path) || ''} 
                        alt="Scatter Plot Visualization" 
                        className="w-full border rounded shadow-sm"
                        onError={(e) => {
                          console.error('Error loading scatter plot image');
                          e.currentTarget.style.display = 'none';
                          const errorDiv = document.createElement('div');
                          errorDiv.className = 'text-sm text-red-500 p-3 border border-red-300 rounded bg-red-50 my-2';
                          errorDiv.textContent = 'Unable to display visualization. The image data may be in an unsupported format.';
                          e.currentTarget.parentNode?.insertBefore(errorDiv, e.currentTarget.nextSibling);
                        }}
                      />
                      {/* Only show path in development mode */}
                      {import.meta.env.DEV && (
                        <div className="text-xs text-gray-500 mt-1">Path: {results.scatter_plot_path?.substring(0, 50)}...</div>
                      )}
                    </>
                  ) : (
                    <div className="text-sm text-red-500">Scatter plot visualization not available</div>
                  )}
                  <p className="text-xs text-gray-500 mt-2">Scatter plot showing outliers (red) vs normal points (blue)</p>
                </div>
              )}
              
              {/* Box Plot - will be available if the backend provides it */}
              {results.box_plot_path && (
                <div className="border rounded-lg p-3 bg-gray-50">
                  <h5 className="text-md font-medium mb-2">Box Plot Analysis</h5>
                  <img 
                    src={getImageSrc(results.box_plot_path) || ''} 
                    alt="Box Plot" 
                    className="w-full border rounded shadow-sm"
                    onError={(e) => {
                      console.error('Error loading box plot image');
                      e.currentTarget.style.display = 'none';
                      const errorDiv = document.createElement('div');
                      errorDiv.className = 'text-sm text-red-500 p-3 border border-red-300 rounded bg-red-50 my-2';
                      errorDiv.textContent = 'Unable to display visualization. The image data may be in an unsupported format.';
                      e.currentTarget.parentNode?.insertBefore(errorDiv, e.currentTarget.nextSibling);
                    }}
                  />
                  {/* Only show path in development mode */}
                  {import.meta.env.DEV && (
                    <div className="text-xs text-gray-500 mt-1">Path: {results.box_plot_path?.substring(0, 50)}...</div>
                  )}
                  <p className="text-xs text-gray-500 mt-2">Box plots showing distribution of features with outliers</p>
                </div>
              )}
            </div>
          </div>
          
          {/* Evaluation Metrics */}
          <div className="mt-6 border-t pt-4">
            <h4 className="font-semibold text-lg mb-3">Evaluation Metrics</h4>
            
            {results.evaluation_metrics ? (
              <div>
                {/* Debug information in development mode */}
                {import.meta.env.DEV && (
                  <div className="mb-4 p-2 bg-gray-100 rounded text-xs font-mono overflow-auto max-h-32">
                    <details>
                      <summary className="cursor-pointer font-semibold">Debug: Raw Metrics Data</summary>
                      <pre>{JSON.stringify(results.evaluation_metrics, null, 2)}</pre>
                    </details>
                  </div>
                )}
                
                {/* Display note if metrics have a note field */}
                {results.evaluation_metrics.note && (
                  <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md text-sm">
                    <div className="flex items-start">
                      <svg className="w-5 h-5 text-yellow-400 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2h-1V9z" clipRule="evenodd" />
                      </svg>
                      <span>{results.evaluation_metrics.note}</span>
                    </div>
                  </div>
                )}
                
                {/* Check if all metrics are null */}
                {results.evaluation_metrics?.precision === null && 
                 results.evaluation_metrics?.recall === null && 
                 results.evaluation_metrics?.f1 === null && 
                 results.evaluation_metrics?.accuracy === null ? (
                  <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
                    <div className="flex items-center text-gray-600">
                      <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <div>
                        <h5 className="font-medium text-blue-700">Evaluation Metrics Not Available</h5>
                        <p className="mt-1 text-sm text-gray-600">
                          Metrics require labeled data (ground truth) to evaluate model performance. 
                          To see evaluation metrics, please upload a file with known outlier labels.
                        </p>
                        <div className="mt-3 flex">
                          <button 
                            className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors"
                            onClick={() => window.open(`${API_BASE_URL}/api/v1/outliers/runs/${results.run_id}/evaluate`, '_blank')}
                          >
                            Upload Ground Truth Data
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="bg-blue-50 p-3 rounded-md shadow-sm">
                      <div className="text-center text-sm font-medium text-gray-700">Precision</div>
                      <div className="text-center text-xl font-semibold text-blue-600">
                        {typeof results.evaluation_metrics?.precision === 'number' 
                          ? results.evaluation_metrics.precision.toFixed(3) 
                          : 'N/A'}
                      </div>
                    </div>
                    <div className="bg-blue-50 p-3 rounded-md shadow-sm">
                      <div className="text-center text-sm font-medium text-gray-700">Recall</div>
                      <div className="text-center text-xl font-semibold text-blue-600">
                        {typeof results.evaluation_metrics?.recall === 'number' 
                          ? results.evaluation_metrics.recall.toFixed(3) 
                          : 'N/A'}
                      </div>
                    </div>
                    <div className="bg-blue-50 p-3 rounded-md shadow-sm">
                      <div className="text-center text-sm font-medium text-gray-700">F1 Score</div>
                      <div className="text-center text-xl font-semibold text-blue-600">
                        {typeof results.evaluation_metrics?.f1 === 'number' 
                          ? results.evaluation_metrics.f1.toFixed(3) 
                          : 'N/A'}
                      </div>
                    </div>
                    <div className="bg-blue-50 p-3 rounded-md shadow-sm">
                      <div className="text-center text-sm font-medium text-gray-700">Accuracy</div>
                      <div className="text-center text-xl font-semibold text-blue-600">
                        {typeof results.evaluation_metrics?.accuracy === 'number' 
                          ? results.evaluation_metrics.accuracy.toFixed(3) 
                          : 'N/A'}
                      </div>
                    </div>
                  </div>
                )}
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                  {typeof results.evaluation_metrics.roc_auc === 'number' && (
                    <div className="bg-blue-50 p-3 rounded-lg text-center shadow-sm">
                      <div className="text-sm text-gray-700 font-medium">ROC AUC</div>
                      <div className="text-lg font-semibold text-blue-600">{results.evaluation_metrics.roc_auc.toFixed(3)}</div>
                    </div>
                  )}
                  
                  {typeof results.evaluation_metrics.average_precision === 'number' && (
                    <div className="bg-blue-50 p-3 rounded-lg text-center shadow-sm">
                      <div className="text-sm text-gray-700 font-medium">Average Precision</div>
                      <div className="text-lg font-semibold text-blue-600">{results.evaluation_metrics.average_precision.toFixed(3)}</div>
                    </div>
                  )}
                </div>
                
                {/* Additional metrics if available */}
                {results.evaluation_metrics.confusion_matrix && (
                  <div className="mb-4">
                    <h5 className="font-medium mb-2">Confusion Matrix</h5>
                    <div className="bg-gray-50 p-3 rounded border">
                      <pre className="text-sm">
                        {JSON.stringify(results.evaluation_metrics.confusion_matrix, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
                
                {/* Display zero metrics explanation if all metrics are zero */}
                {typeof results.evaluation_metrics.precision === 'number' && 
                 typeof results.evaluation_metrics.recall === 'number' && 
                 typeof results.evaluation_metrics.f1_score === 'number' && 
                 typeof results.evaluation_metrics.accuracy === 'number' && 
                 results.evaluation_metrics.precision === 0 && 
                 results.evaluation_metrics.recall === 0 && 
                 results.evaluation_metrics.f1_score === 0 && 
                 results.evaluation_metrics.accuracy === 0 && 
                 !results.evaluation_metrics.note && (
                  <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-md text-sm">
                    <div className="flex items-start">
                      <svg className="w-5 h-5 text-blue-500 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2h-1V9z" clipRule="evenodd" />
                      </svg>
                      <span>
                        <strong>Note:</strong> Evaluation metrics show zeros because ground truth labels are required for evaluation. 
                        To see accurate metrics, please upload labeled data with known outliers.
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="p-4 bg-gray-50 rounded-md border border-gray-200">
                <div className="flex items-center text-gray-600">
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>No evaluation metrics available. Upload labeled data to see metrics.</span>
                </div>
                <p className="mt-2 text-sm text-gray-500">
                  Evaluation metrics require ground truth labels to compare against the model's predictions.
                  You can upload a file with known outlier labels to evaluate the model's performance.
                </p>
              </div>
            )}
          </div>
          
          {/* Outlier Analysis */}
          <div className="mt-6 border-t pt-4">
            <h4 className="font-semibold text-lg mb-3">Outlier Analysis</h4>
            <div className="bg-yellow-50 p-4 rounded-lg">
              <p className="font-medium">Interpretation:</p>
              <ul className="list-disc pl-5 mt-2 space-y-1">
                <li>Check if outliers represent real errors or normal variations in your data</li>
                <li>Review feature importance to understand which variables contribute most to anomalies</li>
                <li>Consider cluster-specific thresholds if outliers are concentrated in specific groups</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default OutlierMLPage;
