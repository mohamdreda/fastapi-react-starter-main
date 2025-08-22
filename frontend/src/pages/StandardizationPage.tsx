import React, { useEffect, useState } from 'react';
import api from '../lib/axios'; 
import { useAuth } from '../context/AuthContext'; 
import { useSearchParams } from 'react-router-dom';

interface Dataset {
  id: number;
  filename: string;
}

const scalingAlgorithms = [
  { value: 'standard', label: 'Standard Scaler (Z-score)' },
  { value: 'robust', label: 'Robust Scaler' },
];

const StandardizationPage: React.FC = () => {
  const { token } = useAuth();
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get('session_id') || localStorage.getItem('active_session_id') || undefined;
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [outputFilename, setOutputFilename] = useState<string>("");
  const [transformationId, setTransformationId] = useState<number | null>(null);

  // Generate default output filename when dataset is selected
  useEffect(() => {
    if (selectedDataset) {
      const dataset = datasets.find(d => d.id.toString() === selectedDataset);
      if (dataset && dataset.filename) {
        // Remove extension and add suffix
        const baseName = dataset.filename.replace(/\.[^/.]+$/, "");
        setOutputFilename(`${baseName}_standardized.csv`);
      }
    }
  }, [selectedDataset, datasets]);
  const [columns, setColumns] = useState<string[]>([]);
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);

  // Configuration State
  const [scalingAlgo, setScalingAlgo] = useState<string>('standard');
  const [withMean, setWithMean] = useState<boolean>(true);
  const [withStd, setWithStd] = useState<boolean>(true);
  const [withCentering, setWithCentering] = useState<boolean>(true);
  const [withScaling, setWithScaling] = useState<boolean>(true);
  const [quantileRange, setQuantileRange] = useState<[number, number]>([25.0, 75.0]);

  // UI State
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  useEffect(() => {
    if (!token) return;
    setLoading(true);
    api.get('/datasets/')
      .then(res => setDatasets(res.data))
      .catch(() => setErrorMsg("Erreur lors du chargement des datasets."))
      .finally(() => setLoading(false));
  }, [token]);

  useEffect(() => {
    if (!selectedDataset || !token) {
      setColumns([]);
      setSelectedColumns([]);
      return;
    }
    setLoading(true);
    setErrorMsg(null);
    
    // Use the new dedicated endpoint to get columns directly from the file
    console.log(`Fetching columns for dataset ID: ${selectedDataset}`);
    
    api.get(`/transformation/dataset-columns/${selectedDataset}`)
      .then(res => {
        console.log('Dataset columns response:', res.data);
        
        const allColumns = res.data.columns || [];
        const numericColumns = res.data.numeric_columns || [];
        
        if (allColumns.length === 0) {
          console.warn('No columns found in the dataset');
          setErrorMsg("Aucune colonne trouvée dans le dataset. Veuillez vous assurer que le dataset contient des données.");
        } else {
          console.log('Found columns:', allColumns);
          console.log('Numeric columns:', numericColumns);
          
          setColumns(allColumns);
          
          if (numericColumns.length > 0) {
            setSelectedColumns(numericColumns);
            setErrorMsg(null);
          } else {
            // If no numeric columns found, don't pre-select any columns
            setSelectedColumns([]);
            setErrorMsg("Aucune colonne numérique trouvée. Veuillez sélectionner manuellement les colonnes à standardiser.");
          }
        }
      })
      .catch(err => {
        console.error('Error fetching dataset columns:', err);
        setErrorMsg("Erreur lors du chargement des colonnes du dataset.");
        
        // Fallback to the old method if the new endpoint fails
        api.get(`/datasets/${selectedDataset}`)
          .then(res => {
            console.log('Fallback - Dataset response:', res.data);
            
            let cols: string[] = [];
            
            // Extract columns from data_types if available
            if (res.data.data_types && typeof res.data.data_types === 'object') {
              cols = Object.keys(res.data.data_types);
              console.log('Extracted columns from data_types:', cols);
            }
            
            // If we have summary_stats, we might find columns there too
            if (cols.length === 0 && res.data.summary_stats && typeof res.data.summary_stats === 'object') {
              cols = Object.keys(res.data.summary_stats);
              console.log('Extracted columns from summary_stats:', cols);
            }
            
            if (cols.length > 0) {
              setColumns(cols);
              setSelectedColumns(cols);
              setErrorMsg(null);
            }
          })
          .catch(fallbackErr => {
            console.error('Fallback also failed:', fallbackErr);
          });
      })
      .finally(() => setLoading(false));
  }, [selectedDataset, token]);

  const handleTransform = async () => {
    if (!selectedDataset) {
      setErrorMsg('Please select a dataset.');
      return;
    }
    if (selectedColumns.length === 0) {
      setErrorMsg('Please select at least one column to scale.');
      return;
    }

    setLoading(true);
    setErrorMsg(null);
    setSuccessMsg(null);
    setDownloadUrl(null);
    setTransformationId(null);

    let methodConfig: any = {
      method: scalingAlgo,
      columns: selectedColumns,
    };

    if (scalingAlgo === 'standard') {
      methodConfig.with_mean = withMean;
      methodConfig.with_std = withStd;
    } else if (scalingAlgo === 'robust') {
      methodConfig.with_centering = withCentering;
      methodConfig.with_scaling = withScaling;
      methodConfig.quantile_range = quantileRange;
    }

    const config = {
      feature_scaling: {
        methods: [methodConfig],
      },
    };

    try {
      const url = `/transformation/transform${sessionId ? `?session_id=${sessionId}` : ''}`;
      const response = await api.post(url, {
        dataset_id: Number(selectedDataset),
        config
      });
      setSuccessMsg('Transformation appliquée avec succès !');
      
      if (response.data && response.data.transformation_id) {
        setTransformationId(response.data.transformation_id);
        // Set the download URL to the proper endpoint
        setDownloadUrl(`/transformation/download/${response.data.transformation_id}`);
      } else {
        console.error("No transformation ID returned from API");
        setErrorMsg("Erreur: ID de transformation non reçu.");
      }
    } catch (error: any) {
      console.error("Error during transformation:", error);
      setErrorMsg("Erreur lors de la transformation. Veuillez réessayer.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!transformationId) {
      setErrorMsg("Erreur: ID de transformation non disponible.");
      return;
    }
    
    try {
      // Use axios to make an authenticated request to download the file
      const response = await api.get(`/transformation/download/${transformationId}`, {
        responseType: 'blob', // Important for file downloads
        headers: {
          Authorization: `Bearer ${token}` // Include the auth token
        }
      });
      
      // Create a blob URL from the response data
      const blob = new Blob([response.data]);
      const url = window.URL.createObjectURL(blob);
      
      // Create a hidden anchor element to trigger the download
      const downloadLink = document.createElement('a');
      downloadLink.href = url;
      downloadLink.download = outputFilename; // Use the custom filename
      document.body.appendChild(downloadLink);
      downloadLink.click();
      
      // Clean up
      window.URL.revokeObjectURL(url);
      document.body.removeChild(downloadLink);
    } catch (error) {
      console.error('Error downloading file:', error);
      setErrorMsg("Erreur lors du téléchargement du fichier. Veuillez réessayer.");
    }
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6">Standardization / Scaling</h1>

      {/* Select Dataset Card */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Select Dataset</h2>
        <select
          className="w-full p-2 border rounded"
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
          disabled={!datasets.length}
        >
          <option value="">Select a dataset</option>
          {datasets.map((ds) => (
            <option key={ds.id} value={ds.id.toString()}>
              {ds.filename}
            </option>
          ))}
        </select>
      </div>

      {/* Scaling Configuration Card */}
      {selectedDataset && (
        <div className="mb-6 bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-3">Scaling Configuration</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
            {/* Scaling Method */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Scaling Method</label>
              <select
                className="w-full p-2 border rounded"
                value={scalingAlgo}
                onChange={(e) => setScalingAlgo(e.target.value)}
                disabled={loading}
              >
                {scalingAlgorithms.map((algo) => (
                  <option key={algo.value} value={algo.value}>
                    {algo.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Columns to Scale */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Columns to Scale</label>
              <select
                multiple
                className="w-full p-2 border rounded"
                value={selectedColumns}
                onChange={(e) =>
                  setSelectedColumns(Array.from(e.target.selectedOptions, (option) => option.value))
                }
                disabled={loading || columns.length === 0}
                size={Math.min(8, columns.length || 1)}
              >
                {columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>
          </div>
          
          {/* Advanced Options */}
          <div className="mt-4 border-t pt-4">
             <h3 className="text-md font-semibold mb-3">Advanced Options for {scalingAlgo === 'standard' ? 'Standard Scaler' : 'Robust Scaler'}</h3>
              {scalingAlgo === 'standard' && (
                <div className="flex items-center gap-6">
                  <label className="flex items-center cursor-pointer">
                    <input type="checkbox" checked={withMean} onChange={() => setWithMean(!withMean)} className="mr-2 h-4 w-4" />
                    Center data (with_mean)
                  </label>
                  <label className="flex items-center cursor-pointer">
                    <input type="checkbox" checked={withStd} onChange={() => setWithStd(!withStd)} className="mr-2 h-4 w-4" />
                    Scale to unit variance (with_std)
                  </label>
                </div>
              )}
              {scalingAlgo === 'robust' && (
                 <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="flex flex-col gap-2">
                         <label className="flex items-center cursor-pointer">
                            <input type="checkbox" checked={withCentering} onChange={() => setWithCentering(!withCentering)} className="mr-2 h-4 w-4" />
                            Center data (with_centering)
                        </label>
                        <label className="flex items-center cursor-pointer">
                            <input type="checkbox" checked={withScaling} onChange={() => setWithScaling(!withScaling)} className="mr-2 h-4 w-4" />
                            Scale data (with_scaling)
                        </label>
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Quantile Range (0-100)</label>
                        <div className="flex items-center gap-2">
                           <input
                                type="number"
                                value={quantileRange[0]}
                                min="0" max="100" step="0.1"
                                className="w-24 p-2 border rounded"
                                onChange={(e) => setQuantileRange([Number(e.target.value), quantileRange[1]])}
                            />
                            <span>-</span>
                            <input
                                type="number"
                                value={quantileRange[1]}
                                min="0" max="100" step="0.1"
                                className="w-24 p-2 border rounded"
                                onChange={(e) => setQuantileRange([quantileRange[0], Number(e.target.value)])}
                            />
                        </div>
                    </div>
                 </div>
              )}
          </div>


          <div className="mt-6">
            <button
              onClick={handleTransform}
              disabled={loading}
              className={`w-full px-4 py-2 rounded text-white font-semibold transition-colors ${
                loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {loading ? 'Processing...' : 'Run Standardization'}
            </button>
          </div>
        </div>
      )}

      {/* Messages */}
      {errorMsg && (
        <div className="my-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {errorMsg}
        </div>
      )}

      {successMsg && (
        <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative mb-4">
          {successMsg}
        </div>
      )}

      {downloadUrl && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded">
          <div className="mb-4">
            <label htmlFor="outputFilename" className="block text-sm font-medium text-gray-700 mb-1">
              Output Filename
            </label>
            <input
              type="text"
              id="outputFilename"
              value={outputFilename}
              onChange={(e) => setOutputFilename(e.target.value)}
              className="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-gray-300 rounded-md"
              placeholder="Enter filename for the transformed data"
            />
          </div>
          <button
            onClick={handleDownload}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded inline-flex items-center"
          >
            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
            </svg>
            Download Transformed File
          </button>
        </div>
      )}
    </div>
  );
};

export default StandardizationPage;