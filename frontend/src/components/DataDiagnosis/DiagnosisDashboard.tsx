import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Plot from 'react-plotly.js';
import { useAuth } from '@/context/AuthContext';
import { useAppState } from '@/context/AppContext';

const API_BASE = import.meta.env.VITE_API_URL;

interface Dataset {
  id: number;
  filename: string;
  created_at: string;
}

interface MissingValuesAnalysis {
  total_missing: number;
  missing_percentage: number;
  per_column: Record<string, { count: number; percentage: number }>;
}

interface DiagnosisData {
  id: number;
  filename: string;
  created_at: string;
  analysis: {
    missing_values: MissingValuesAnalysis;
    duplicates: {
      count: number;
      details: Record<string, any>;
    };
    categorical_issues: Record<string, any>;
    summary_stats: Record<string, any>;
    data_types: Record<string, any>;
    id_columns: string[];
    type_issues: Record<string, any>;
  };
  visualizations: Array<{
    type: string;
    endpoint: string;
    category: string;
  }>;
}

export const DiagnosisDashboard: React.FC = () => {
  const { theme } = useAppState();
  const { user, token } = useAuth();
  const { datasetId: routeDatasetId } = useParams<{ datasetId?: string }>();
  const navigate = useNavigate();

  const [datasets, setDatasets] = React.useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = React.useState<string>('');
  const [diagnosisData, setDiagnosisData] = React.useState<DiagnosisData | null>(null);
  const [visualizations, setVisualizations] = React.useState<Record<string, any>>({});
  const [loading, setLoading] = React.useState<boolean>(false);
  const [error, setError] = React.useState<string | null>(null);
  const [activeTab, setActiveTab] = React.useState<string>('basic');
  const [failedVisualizations, setFailedVisualizations] = React.useState<string[]>([]);
  const [retryCount, setRetryCount] = React.useState<number>(0);
  const safeJsonParse = async (response: Response) => {
    try {
      const text = await response.text();
      try {
        return JSON.parse(text);
      } catch (error) {
        const jsonErr = error as Error;
        console.error('Error parsing JSON:', jsonErr, 'Response text:', text.substring(0, 200));
        throw new Error(`Invalid JSON response: ${jsonErr.message}`);
      }
    } catch (err) {
      console.error('Error reading response text:', err);
      throw new Error('Failed to read response');
    }
  };

  React.useEffect(() => {
    if (!routeDatasetId) {
      const fetchDatasets = async () => {
        try {
          const res = await fetch(`${API_BASE}/api/v1/datasets`, {
            headers: { Authorization: `Bearer ${token}` },
          });

          if (!res.ok) {
            const errorText = await res.text();
            throw new Error(`HTTP ${res.status}: ${errorText.slice(0, 100)}`);
          }

          const data = await safeJsonParse(res);
          setDatasets(data);
          setError(null);
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to load datasets');
          console.error('Dataset fetch error:', err);
        }
      };
      fetchDatasets();
    }
  }, [routeDatasetId, token]);

  // Function to check if token is expired
  const isTokenExpired = (token: string | null): boolean => {
    if (!token) return true;

    try {
      // Decode the token (JWT format: header.payload.signature)
      const base64Url = token.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const payload = JSON.parse(window.atob(base64));

      // Check if token is expired
      const exp = payload.exp * 1000; // Convert to milliseconds
      return Date.now() >= exp;
    } catch (e) {
      console.error('Error checking token expiration:', e);
      return true; // Assume expired if there's an error
    }
  };

  React.useEffect(() => {
    if (routeDatasetId && diagnosisData) {
      // Fetch visualizations for each type
      const fetchVisualizations = async () => {
        setFailedVisualizations([]);

        // Check if token exists and is not expired
        if (!token) {
          setError('No authentication token found. Please log in again.');
          return;
        }

        if (isTokenExpired(token)) {
          setError('Your session has expired. Please log in again.');
          return;
        }

        // Create a queue of visualization types to fetch
        const vizQueue = [...diagnosisData.visualizations];

        // Process visualizations sequentially to avoid overwhelming the browser
        // No need for batch processing since we're doing one at a time

        // Process each visualization one at a time with delays
        for (let i = 0; i < vizQueue.length; i++) {
          const viz = vizQueue[i];

          try {
            // Add a delay between requests
            await new Promise(resolve => setTimeout(resolve, 1000));

            console.log(`[DEBUG] Fetching visualization: ${viz.type} from ${API_BASE}${viz.endpoint}`);

            // Simple direct fetch approach
            console.log(`[DEBUG] Fetching visualization: ${viz.type} from ${API_BASE}${viz.endpoint}`);
            
            try {
              // Use standard fetch with minimal options
              const res = await fetch(`${API_BASE}${viz.endpoint}`, {
                method: 'GET',
                headers: { 
                  Authorization: `Bearer ${token}`
                }
              });
              
              if (!res.ok) {
                const errorText = await res.text();
                console.error(`[ERROR] HTTP ${res.status} for ${viz.type}:`, errorText.slice(0, 100));
                throw new Error(`HTTP ${res.status}: ${errorText.slice(0, 100)}`);
              }
              
              const data = await safeJsonParse(res);
              
              if (!data || !data.data) {
                console.error(`[ERROR] Invalid data format for ${viz.type}:`, data);
                throw new Error(`Invalid data format for ${viz.type}`);
              }
              
              console.log(`[SUCCESS] Loaded ${viz.type} visualization`);
              setVisualizations(prev => ({
                ...prev,
                [viz.type]: data.data
              }));
            } catch (error) {
              console.error(`Error fetching ${viz.type} visualization:`, error);
              setFailedVisualizations(prev => [...prev, viz.type]);
              setVisualizations(prev => ({
                ...prev,
                [viz.type]: { error: error instanceof Error ? error.message : `Failed to load ${viz.type} visualization` }
              }));
            }

            // This code is unreachable due to the try/catch block above
            // It has been moved into the try block
          } catch (err) {
            console.error(`Error fetching ${viz.type} visualization:`, err);

            // Add to failed visualizations
            if (!failedVisualizations.includes(viz.type)) {
              setFailedVisualizations(prev => [...prev, viz.type]);
            }

            // Update visualizations with error
            setVisualizations(prev => ({
              ...prev,
              [viz.type]: { error: err instanceof Error ? err.message : `Failed to load ${viz.type} visualization` }
            }));
          }

          // Add a longer delay after each visualization
          await new Promise(resolve => setTimeout(resolve, 1500));
        }
      };

      fetchVisualizations();
    }
  }, [routeDatasetId, diagnosisData, token, retryCount]);

  // Function to retry failed visualizations
  const retryFailedVisualizations = async () => {
    // Make a copy of the current failed visualizations
    const vizToRetry = [...failedVisualizations];
    
    // Clear the failed visualizations list before retrying
    setFailedVisualizations([]);
    
    // Process retries sequentially with delays
    for (const vizType of vizToRetry) {
      console.log(`Retrying visualization: ${vizType}`);
      await fetchVisualization(vizType);
      // Add a delay between retries
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
    setRetryCount(prevCount => prevCount + 1);
  };

  // Track which visualizations are currently being fetched to prevent duplicate requests
  const [fetchingVisualizations, setFetchingVisualizations] = React.useState<string[]>([]);

  // Function to fetch a specific visualization
  const fetchVisualization = React.useCallback(async (vizType: string) => {
    if (!diagnosisData) return;

    // Prevent duplicate requests for the same visualization
    if (fetchingVisualizations.includes(vizType)) {
      console.log(`Already fetching ${vizType}, skipping duplicate request`);
      return;
    }

    try {
      // Mark this visualization as being fetched
      setFetchingVisualizations(prev => [...prev, vizType]);
      setError(null);

      // Check if token exists and is not expired
      if (!token) {
        setError('No authentication token found. Please log in again.');
        return;
      }

      if (isTokenExpired(token)) {
        setError('Your session has expired. Please log in again.');
        return;
      }

      const endpoint = diagnosisData.visualizations.find(v => v.type === vizType)?.endpoint;
      if (!endpoint) {
        console.error(`No endpoint found for visualization type: ${vizType}`);
        return;
      }

      console.log(`Fetching visualization: ${vizType} from ${endpoint}`);

      // Simple direct fetch approach
      console.log(`[DEBUG] Fetching visualization: ${vizType} from ${API_BASE}${endpoint}`);
      
      try {
        // Use standard fetch with minimal options
        const res = await fetch(`${API_BASE}${endpoint}`, {
          method: 'GET',
          headers: { 
            Authorization: `Bearer ${token}`
          }
        });
        
        if (!res.ok) {
          const errorText = await res.text();
          console.error(`[ERROR] HTTP ${res.status} for ${vizType}:`, errorText.slice(0, 100));
          throw new Error(`HTTP ${res.status}: ${errorText.slice(0, 100)}`);
        }
        
        const data = await safeJsonParse(res);
        
        if (!data || !data.data) {
          console.error(`[ERROR] Invalid data format for ${vizType}:`, data);
          throw new Error(`Invalid data format for ${vizType}`);
        }
        
        console.log(`[SUCCESS] Loaded ${vizType} visualization`);
        setVisualizations(prev => ({
          ...prev,
          [vizType]: data.data
        }));
        
        // Remove from failed visualizations if it was there
        setFailedVisualizations(prev => prev.filter(v => v !== vizType));
        
        // Remove from fetching visualizations
        setFetchingVisualizations(prev => prev.filter(v => v !== vizType));
        
        return;
      } catch (error) {
        console.error(`Error fetching ${vizType} visualization:`, error);
        
        // Add to failed visualizations if it's not already there
        if (!failedVisualizations.includes(vizType)) {
          setFailedVisualizations(prev => [...prev, vizType]);
        }
        
        // Update visualizations with error
        setVisualizations(prev => ({
          ...prev,
          [vizType]: { error: error instanceof Error ? error.message : `Failed to load ${vizType} visualization` }
        }));
        
        // Remove from fetching visualizations
        setFetchingVisualizations(prev => prev.filter(v => v !== vizType));
        
        return;
      }
    } catch (err) {
      console.error(`Error fetching ${vizType} visualization:`, err);

      // Add to failed visualizations
      if (!failedVisualizations.includes(vizType)) {
        setFailedVisualizations(prev => [...prev, vizType]);
      }

      // Update visualizations with error
      setVisualizations(prev => ({
        ...prev,
        [vizType]: { error: err instanceof Error ? err.message : `Failed to load ${vizType} visualization` }
      }));
    } finally {
      // Remove this visualization from the fetching list
      setFetchingVisualizations(prev => prev.filter(v => v !== vizType));
    }
  }, [diagnosisData, token, failedVisualizations, fetchingVisualizations]);

  const fetchDiagnosisData = React.useCallback(async () => {
    if (!routeDatasetId) return;

    setLoading(true);
    setError(null);
    setFailedVisualizations([]);

    // Check if token exists and is not expired
    if (!token) {
      setError('No authentication token found. Please log in again.');
      setLoading(false);
      return;
    }

    if (isTokenExpired(token)) {
      setError('Your session has expired. Please log in again.');
      setLoading(false);
      return;
    }

    // Log token info for debugging
    console.log(`[DEBUG] Using token: ${token.substring(0, 15)}...`);
    console.log(`[DEBUG] API URL: ${API_BASE}/api/v1/diagnosis/${routeDatasetId}`);

    try {
      const diagnosisRes = await fetch(
        `${API_BASE}/api/v1/diagnosis/${routeDatasetId}`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );

      if (!diagnosisRes.ok) {
        const errorText = await diagnosisRes.text();
        throw new Error(`HTTP ${diagnosisRes.status}: ${errorText.slice(0, 100)}`);
      }

      const diagnosisData = await safeJsonParse(diagnosisRes);
      setDiagnosisData(diagnosisData);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
      console.error('Diagnosis fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [routeDatasetId, token, safeJsonParse, isTokenExpired]);

  React.useEffect(() => {
    fetchDiagnosisData();
    // Intentionally exclude fetchDiagnosisData from dependencies
    // to prevent infinite loop
  }, [routeDatasetId, token]);

  // Dataset selection UI
  if (!routeDatasetId) {
    return (
      <div className="flex-1 p-6">
        <div className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-6 text-gray-900 dark:text-white">
          <h2 className="text-xl font-semibold mb-4">Select Dataset for Analysis</h2>

          {error && (
            <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4">
              <p className="font-medium">Error:</p>
              <p>{error}</p>
            </div>
          )}

          {failedVisualizations.length > 0 && (
            <div className="bg-yellow-50 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-4">
              <p className="font-medium">Some visualizations failed to load:</p>
              <ul className="list-disc ml-5 mt-2">
                {failedVisualizations.map((vizType: string) => (
                  <li key={vizType}>{vizType} visualization</li>
                ))}
              </ul>
              <button
                onClick={retryFailedVisualizations}
                className="mt-3 bg-yellow-500 hover:bg-yellow-600 text-white py-1 px-4 rounded text-sm font-medium"
              >
                Retry Failed Visualizations
              </button>
            </div>
          )}

          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="block w-full p-2 border rounded-md mb-4 focus:ring-2 focus:ring-primary-500"
          >
            <option value="">Choose a dataset</option>
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.id}>
                {ds.filename} ({new Date(ds.created_at).toLocaleDateString()})
              </option>
            ))}
          </select>

          <button
            disabled={!selectedDataset}
            onClick={() => navigate(`/user/dashboard/${user?.id}/diagnosis/${selectedDataset}`)}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-md font-medium shadow-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700 mt-4"
          >
            Run Analysis
          </button>
        </div>
      </div>
    );
  }

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center p-6">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="p-6 bg-white dark:bg-gray-800 text-gray-900 dark:text-white">
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-md">
          <strong>Error:</strong> {error}
        </div>
        <button
          onClick={() => navigate(`/user/dashboard/${user?.id}/diagnosis`)}
          className="mt-4 px-4 py-2 bg-gray-100 rounded-md hover:bg-gray-200"
        >
          Back to Dataset Selection
        </button>
      </div>
    );
  }

  // Main diagnosis dashboard
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Main content */}
      <main className="p-6 space-y-6">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-4 text-gray-900 dark:text-white">
          <h1 className="text-2xl font-semibold text-gray-800">
            Analysis for {diagnosisData?.filename}
          </h1>
          <p className="text-gray-500 mt-1">
            Uploaded: {diagnosisData?.created_at ?
              new Date(diagnosisData.created_at).toLocaleDateString() :
              'Unknown date'}
          </p>
        </div>

        {/* Quality metrics */}
        <section id="quality" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-4 text-gray-900 dark:text-white">
            <h3 className="text-sm font-medium text-gray-500">Missing Values</h3>
            <p className="mt-2 text-3xl font-semibold text-gray-900">
              {diagnosisData?.analysis?.missing_values?.missing_percentage ? diagnosisData.analysis.missing_values.missing_percentage.toFixed(1) : '0'}%
            </p>
            <p className="text-sm text-gray-500 mt-1">
              Total missing: {diagnosisData?.analysis?.missing_values?.total_missing || 0}
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-4 text-gray-900 dark:text-white">
            <h3 className="text-sm font-medium text-gray-500">Duplicate Rows</h3>
            <p className="mt-2 text-3xl font-semibold text-gray-900">
              {diagnosisData?.analysis?.duplicates?.count || 0}
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-4 text-gray-900 dark:text-white">
            <h3 className="text-sm font-medium text-gray-500">Categorical Columns</h3>
            <p className="mt-2 text-3xl font-semibold text-gray-900">
              {Object.keys(diagnosisData?.analysis?.categorical_issues || {}).length}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              With potential issues
            </p>
          </div>

        </section>

        {/* Visualization Tabs */}
        <div className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-6 text-gray-900 dark:text-white">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8" aria-label="Tabs">
              <button
                onClick={() => setActiveTab('basic')}
                className={`${activeTab === 'basic' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
              >
                Basic Analysis
              </button>
              <button
                onClick={() => setActiveTab('advanced')}
                className={`${activeTab === 'advanced' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'} whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
              >
                Advanced Analysis
              </button>
            </nav>
          </div>

          {/* Basic Visualizations Tab */}
          {activeTab === 'basic' && (
            <div className="p-4 space-y-6">
              {/* Missing Values */}
              {visualizations.missing && !visualizations.missing.error && visualizations.missing.plot && (
                <section className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-4 text-gray-900 dark:text-white">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Missing Values Distribution</h3>
                  <Plot
                    data={visualizations.missing.plot.data}
                    layout={{
                      ...visualizations.missing.plot.layout,
                      height: 400,
                      margin: { t: 40, r: 20, b: 60, l: 60 },
                      plot_bgcolor: '#fff',
                      paper_bgcolor: '#fff',
                    }}
                    config={{
                      responsive: true,
                      displayModeBar: true
                    }}
                    className="w-full"
                  />
                </section>
              )}
              {visualizations.missing && visualizations.missing.error && (
                <section className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-4 text-gray-900 dark:text-white">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Missing Values Distribution</h3>
                  <div className="p-4 bg-red-50 border border-red-200 rounded">
                    <p className="text-red-600">Error loading visualization: {visualizations.missing.error}</p>
                  </div>
                </section>
              )}

              {/* Duplicates */}
              {visualizations.duplicates && !visualizations.duplicates.error && (
                <section className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-4 text-gray-900 dark:text-white">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Duplicate Rows Analysis</h3>
                  
                  {/* Display pie chart if available */}
                  {visualizations.duplicates.additional_plots && visualizations.duplicates.additional_plots.pie_chart ? (
                    <Plot
                      data={visualizations.duplicates.additional_plots.pie_chart.data}
                      layout={{
                        ...visualizations.duplicates.additional_plots.pie_chart.layout,
                        height: 400,
                        margin: { t: 40, r: 20, b: 60, l: 60 },
                        plot_bgcolor: '#fff',
                        paper_bgcolor: '#fff',
                      }}
                      config={{
                        responsive: true,
                        displayModeBar: true
                      }}
                      className="w-full"
                    />
                  ) : (
                    <div className="p-4 bg-yellow-50 border border-yellow-200 rounded">
                      <p className="text-yellow-600">Pie chart visualization not available</p>
                    </div>
                  )}
                </section>
              )}
              {visualizations.duplicates && visualizations.duplicates.error && (
                <section className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-4 text-gray-900 dark:text-white">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Duplicate Rows Analysis</h3>
                  <div className="p-4 bg-red-50 border border-red-200 rounded">
                    <p className="text-red-600">Error loading visualization: {visualizations.duplicates.error}</p>
                  </div>
                </section>
              )}

              {/* Categorical */}
              {visualizations.categorical && !visualizations.categorical.error && visualizations.categorical.plot && (
                <section className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-4 text-gray-900 dark:text-white">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Categorical Data Analysis</h3>
                  <Plot
                    data={visualizations.categorical.plot.data}
                    layout={{
                      ...visualizations.categorical.plot.layout,
                      height: 400,
                      margin: { t: 40, r: 20, b: 60, l: 60 },
                      plot_bgcolor: '#fff',
                      paper_bgcolor: '#fff',
                    }}
                    config={{
                      responsive: true,
                      displayModeBar: true
                    }}
                    className="w-full"
                  />
                </section>
              )}
              {visualizations.categorical && visualizations.categorical.error && (
                <section className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-4 text-gray-900 dark:text-white">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Categorical Data Analysis</h3>
                  <div className="p-4 bg-red-50 border border-red-200 rounded">
                    <p className="text-red-600">Error loading visualization: {visualizations.categorical.error}</p>
                  </div>
                </section>
              )}

              {/* Outliers */}
              {visualizations.outliers && !visualizations.outliers.error && visualizations.outliers.plot && visualizations.outliers.plot.data && visualizations.outliers.plot.layout && (
                <section className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-4 text-gray-900 dark:text-white">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Outlier Detection</h3>
                  <Plot
                    data={visualizations.outliers.plot.data}
                    layout={{
                      ...(visualizations.outliers.plot.layout || {}),
                      height: 500,
                      margin: { t: 40, r: 20, b: 60, l: 60 },
                      plot_bgcolor: '#fff',
                      paper_bgcolor: '#fff',
                    }}
                    config={{
                      responsive: true,
                      displayModeBar: true
                    }}
                    className="w-full"
                  />
                </section>
              )}
            </div>
          )}

          {/* Advanced Visualizations Tab */}
          {activeTab === 'advanced' && (
            <div className="p-4 space-y-6">
              {/* Dataset Structure */}
              {visualizations.structure && (
                <section className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-4 text-gray-900 dark:text-white">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Dataset Structure</h3>
                  {visualizations.structure.error ? (
                    <div className="p-4 bg-red-50 border border-red-200 rounded">
                      <p className="text-red-600">Error loading visualization: {visualizations.structure.error}</p>
                      <button
                        onClick={() => fetchVisualization('structure')}
                        className="mt-2 px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
                      >
                        Retry Loading Structure
                      </button>
                    </div>
                  ) : (
                    visualizations.structure.plot && visualizations.structure.plot.data && visualizations.structure.plot.layout && (
                      <>
                        <Plot
                          data={visualizations.structure.plot.data}
                          layout={{
                            ...(visualizations.structure.plot.layout || {}),
                            height: 400,
                            margin: { t: 40, r: 20, b: 60, l: 60 },
                            plot_bgcolor: '#fff',
                            paper_bgcolor: '#fff',
                          }}
                          config={{
                            responsive: true,
                            displayModeBar: true
                          }}
                          className="w-full"
                        />
                        {visualizations.structure.stats && (
                          <div className="mt-4">
                            <h4 className="text-md font-medium text-gray-700">Dataset Information</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-4 text-gray-900 dark:text-white">
                              <div className="text-center p-4">
                                <p><span className="font-medium">Rows:</span> {visualizations.structure.stats.shape?.[0]}</p>
                                <p><span className="font-medium">Columns:</span> {visualizations.structure.stats.shape?.[1]}</p>
                              </div>
                            </div>
                          </div>
                        )}
                      </>
                    )
                  )}
                </section>
              )}
              {visualizations.structure && visualizations.structure.error && (
                <section className="bg-white rounded-lg border border-gray-200 p-4">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Dataset Structure</h3>
                  <div className="p-4 bg-red-50 border border-red-200 rounded">
                    <p className="text-red-600">Error loading visualization: {visualizations.structure.error}</p>
                    <button
                      onClick={() => fetchVisualization('structure')}
                      className="mt-2 px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
                    >
                      Retry Loading Structure
                    </button>
                  </div>
                </section>
              )}

              {/* Distribution Analysis */}
              {visualizations.distribution && !visualizations.distribution.error && visualizations.distribution.plots && (
                <section className="bg-white rounded-lg border border-gray-200 p-4">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Data Distribution Analysis</h3>
                  {visualizations.distribution.plots?.histograms && Object.entries(visualizations.distribution.plots.histograms).map(([col, plot]) => {
                    if (!(plot as any)?.data || !(plot as any)?.layout) return null;
                    return (
                      <div key={col} className="mb-6">
                        <h4 className="text-md font-medium text-gray-700 mb-2">Numerical: {col}</h4>
                        <Plot
                          data={(plot as any).data}
                          layout={{
                            ...((plot as any).layout || {}),
                            height: 300,
                            margin: { t: 40, r: 20, b: 60, l: 60 },
                            plot_bgcolor: '#fff',
                            paper_bgcolor: '#fff',
                          }}
                          config={{
                            responsive: true,
                            displayModeBar: true
                          }}
                          className="w-full"
                        />
                      </div>
                    );
                  })}
                  {visualizations.distribution.plots?.bar_charts && Object.entries(visualizations.distribution.plots.bar_charts).map(([col, plot]) => {
                    if (!(plot as any)?.data || !(plot as any)?.layout) return null;
                    return (
                      <div key={col} className="mb-6">
                        <h4 className="text-md font-medium text-gray-700 mb-2">Categorical: {col}</h4>
                        <Plot
                          data={(plot as any).data}
                          layout={{
                            ...((plot as any).layout || {}),
                            height: 300,
                            margin: { t: 40, r: 20, b: 60, l: 60 },
                            plot_bgcolor: '#fff',
                            paper_bgcolor: '#fff',
                          }}
                          config={{
                            responsive: true,
                            displayModeBar: true
                          }}
                          className="w-full"
                        />
                      </div>
                    );
                  })}
                </section>
              )}
              {visualizations.distribution && visualizations.distribution.error && (
                <section className="bg-white rounded-lg border border-gray-200 p-4">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Data Distribution Analysis</h3>
                  <div className="p-4 bg-red-50 border border-red-200 rounded">
                    <p className="text-red-600">Error loading visualization: {visualizations.distribution.error}</p>
                    <button
                      onClick={() => fetchVisualization('distribution')}
                      className="mt-2 px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
                    >
                      Retry Loading Distribution
                    </button>
                  </div>
                </section>
              )}

              {/* Correlation Analysis */}
              {visualizations.correlation && !visualizations.correlation.error && visualizations.correlation.plots && (
                <section className="bg-white rounded-lg border border-gray-200 p-4">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Correlation Analysis</h3>
                  {visualizations.correlation.plots?.heatmap && visualizations.correlation.plots.heatmap.data && visualizations.correlation.plots.heatmap.layout && (
                    <div className="mb-6">
                      <h4 className="text-md font-medium text-gray-700 mb-2">Correlation Heatmap</h4>
                      <Plot
                        data={visualizations.correlation.plots.heatmap.data}
                        layout={{
                          ...(visualizations.correlation.plots.heatmap.layout || {}),
                          height: 500,
                          margin: { t: 40, r: 20, b: 60, l: 60 },
                          plot_bgcolor: '#fff',
                          paper_bgcolor: '#fff',
                        }}
                        config={{
                          responsive: true,
                          displayModeBar: true
                        }}
                        className="w-full"
                      />
                    </div>
                  )}
                  {visualizations.correlation.plots?.scatter && visualizations.correlation.plots.scatter.data && visualizations.correlation.plots.scatter.layout && (
                    <div className="mb-6">
                      <h4 className="text-md font-medium text-gray-700 mb-2">Scatter Plot of Strongest Correlation</h4>
                      <Plot
                        data={visualizations.correlation.plots.scatter.data}
                        layout={{
                          ...(visualizations.correlation.plots.scatter.layout || {}),
                          height: 400,
                          margin: { t: 40, r: 20, b: 60, l: 60 },
                          plot_bgcolor: '#fff',
                          paper_bgcolor: '#fff',
                        }}
                        config={{
                          responsive: true,
                          displayModeBar: true
                        }}
                        className="w-full"
                      />
                    </div>
                  )}
                </section>
              )}
              {visualizations.correlation && visualizations.correlation.error && (
                <section className="bg-white rounded-lg border border-gray-200 p-4">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Correlation Analysis</h3>
                  <div className="p-4 bg-red-50 border border-red-200 rounded">
                    <p className="text-red-600">Error loading visualization: {visualizations.correlation.error}</p>
                    <button
                      onClick={() => fetchVisualization('correlation')}
                      className="mt-2 px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
                    >
                      Retry Loading Correlation
                    </button>
                  </div>
                </section>
              )}

              {/* Summary Statistics */}
              {visualizations.summary && !visualizations.summary.error && visualizations.summary.plot && (
                <section className="bg-white rounded-lg border border-gray-200 p-4">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Summary Statistics</h3>
                  <Plot
                    data={visualizations.summary.plot.data}
                    layout={{
                      ...visualizations.summary.plot.layout,
                      height: 400,
                      margin: { t: 40, r: 20, b: 60, l: 60 },
                      plot_bgcolor: '#fff',
                      paper_bgcolor: '#fff',
                    }}
                    config={{
                      responsive: true,
                      displayModeBar: true
                    }}
                    className="w-full"
                  />
                </section>
              )}
              {visualizations.summary && visualizations.summary.error && (
                <section className="bg-white rounded-lg border border-gray-200 p-4">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Summary Statistics</h3>
                  <div className="p-4 bg-red-50 border border-red-200 rounded">
                    <p className="text-red-600">Error loading visualization: {visualizations.summary.error}</p>
                    <button
                      onClick={() => fetchVisualization('summary')}
                      className="mt-2 px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
                    >
                      Retry Loading Summary
                    </button>
                  </div>
                </section>
              )}

              {/* Categorical Consistency */}
              {visualizations.categorical_consistency && !visualizations.categorical_consistency.error && (
                <section className="bg-white rounded-lg border border-gray-200 p-4">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Categorical Consistency Check</h3>
                  {visualizations.categorical_consistency.plots && Object.entries(visualizations.categorical_consistency.plots).map(([col, plot]) => (
                    <div key={col} className="mb-6">
                      <h4 className="text-md font-medium text-gray-700 mb-2">Column: {col} (Red = Potential Issues)</h4>
                      <Plot
                        data={(plot as any).data}
                        layout={{
                          ...(plot as any).layout,
                          height: 300,
                          margin: { t: 40, r: 20, b: 60, l: 60 },
                          plot_bgcolor: '#fff',
                          paper_bgcolor: '#fff',
                        }}
                        config={{
                          responsive: true,
                          displayModeBar: true
                        }}
                        className="w-full"
                      />
                    </div>
                  ))}
                </section>
              )}
              {visualizations.categorical_consistency && visualizations.categorical_consistency.error && (
                <section className="bg-white rounded-lg border border-gray-200 p-4">
                  <h3 className="text-lg font-medium mb-4 text-gray-800">Categorical Consistency Check</h3>
                  <div className="p-4 bg-red-50 border border-red-200 rounded">
                    <p className="text-red-600">Error loading visualization: {visualizations.categorical_consistency.error}</p>
                    <button 
                      onClick={() => fetchVisualization('categorical_consistency')} 
                      className="mt-2 px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
                    >
                      Retry Loading Categorical Consistency
                    </button>
                  </div>
                </section>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};