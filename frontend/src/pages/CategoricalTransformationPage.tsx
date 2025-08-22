import React, { useEffect, useState } from 'react';
import api from '../lib/axios';
import { useAuth } from '../context/AuthContext';
import { useSearchParams } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  RadioGroup,
  FormControlLabel,
  Radio,
  Divider,
  Alert
} from '@mui/material';

const categoricalAlgorithms = [
  { value: 'label', label: 'Label Encoding' },
  { value: 'one_hot', label: 'OneHot Encoding' }
];

interface Dataset {
  id: number;
  filename: string;
  file_type: string;
  created_at: string;
}

interface DatasetDetails extends Dataset {
  data_types?: Record<string, string>;
}

const CategoricalTransformationPage: React.FC = () => {
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
        setOutputFilename(`${baseName}_encoded.csv`);
      }
    }
  }, [selectedDataset, datasets]);
  const [catAlgo, setCatAlgo] = useState<string>('label');
  const [loading, setLoading] = useState<boolean>(false);
  const [successMsg, setSuccessMsg] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [columns, setColumns] = useState<string[]>([]);
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [dropFirst, setDropFirst] = useState<boolean>(false);
  const { token } = useAuth();

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
    
    // First try the new endpoint that reads columns directly from the dataset file
    api.get(`/transformation/dataset-columns/${selectedDataset}`)
      .then(res => {
        console.log('Dataset columns response:', res.data);
        if (res.data && Array.isArray(res.data.columns)) {
          // Filter to only categorical columns (non-numeric)
          const allColumns = res.data.columns;
          const numericColumns = res.data.numeric_columns || [];
          // Get categorical columns by excluding numeric columns
          const categoricalColumns = allColumns.filter(col => !numericColumns.includes(col));
          
          setColumns(categoricalColumns);
          console.log('Categorical columns:', categoricalColumns);
          setSelectedColumns([]);
        } else {
          throw new Error('No columns found in response');
        }
      })
      .catch(err => {
        console.error('Error fetching columns from dataset-columns endpoint:', err);
        // Fallback to the old method using dataset metadata
        api.get(`/datasets/${selectedDataset}`)
          .then(res => {
            let cols: string[] = [];
            if (Array.isArray(res.data.columns)) {
              cols = res.data.columns;
            } else if (res.data.data_types && typeof res.data.data_types === 'object') {
              cols = Object.keys(res.data.data_types);
            }
            console.log('Fallback columns from dataset metadata:', cols);
            setColumns(cols);
            setSelectedColumns([]);
          })
          .catch(fallbackErr => {
            console.error('Error in fallback column loading:', fallbackErr);
            setErrorMsg("Erreur lors du chargement des colonnes du dataset.");
          });
      })
      .finally(() => setLoading(false));
  }, [selectedDataset, token]);

  const handleTransform = async () => {
    if (!selectedDataset) {
      setErrorMsg('Veuillez choisir un dataset.');
      return;
    }
    if (selectedColumns.length === 0) {
      setErrorMsg('Veuillez sélectionner au moins une colonne.');
      return;
    }
    setLoading(true);
    setSuccessMsg('');
    setErrorMsg('');
    setDownloadUrl(null);
    setTransformationId(null);
    
    try {
      const methodConfig: any = {
        method: catAlgo,
        columns: selectedColumns
      };
      if (catAlgo === 'one_hot') {
        methodConfig.drop = dropFirst ? 'first' : undefined;
      }

      const config = {
        categorical_encoding: {
          methods: [methodConfig]
        }
      };

      const url = `/transformation/transform${sessionId ? `?session_id=${sessionId}` : ''}`;
      const response = await api.post(url, {
        dataset_id: parseInt(selectedDataset),
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
      setErrorMsg(error?.response?.data?.detail || 'Erreur lors de la transformation. Veuillez réessayer.');
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
      <h1 className="text-3xl font-bold mb-6">Categorical Encoding</h1>

      {/* Select Dataset Card */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Select Dataset</h2>
        <select
          className="w-full p-2 border rounded"
          value={selectedDataset}
          onChange={e => setSelectedDataset(e.target.value)}
          disabled={!datasets.length}
        >
          <option value="">Select a dataset</option>
          {datasets.map(ds => (
            <option key={ds.id} value={ds.id.toString()}>{ds.filename}</option>
          ))}
        </select>
      </div>

      {/* Encoding Configuration Card */}
      {selectedDataset && (
        <div className="mb-6 bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-semibold mb-3">Encoding Configuration</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-4">
            {/* Encoding Method */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Encoding Method</label>
              <select
                className="w-full p-2 border rounded"
                value={catAlgo}
                onChange={e => setCatAlgo(e.target.value)}
                disabled={loading}
              >
                {categoricalAlgorithms.map(algo => (
                  <option key={algo.value} value={algo.value}>{algo.label}</option>
                ))}
              </select>
            </div>

            {/* Columns to Encode */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Columns to Encode</label>
              <select
                multiple
                className="w-full p-2 border rounded"
                value={selectedColumns}
                onChange={e => setSelectedColumns(Array.from(e.target.selectedOptions, option => option.value))}
                disabled={loading || columns.length === 0}
                size={Math.min(8, columns.length || 1)}
              >
                {columns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Advanced Options */}
          {catAlgo === 'one_hot' && (
            <div className="mt-4 border-t pt-4">
              <h3 className="text-md font-semibold mb-3">Advanced Options for OneHot Encoding</h3>
              <label className="flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  id="dropFirst"
                  checked={dropFirst}
                  onChange={() => setDropFirst(!dropFirst)}
                  className="mr-2 h-4 w-4"
                />
                Drop first category (to avoid collinearity)
              </label>
            </div>
          )}

          <div className="mt-6">
            <button
              onClick={handleTransform}
              disabled={loading}
              className={`w-full px-4 py-2 rounded text-white font-semibold transition-colors ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}
            >
              {loading ? 'Processing...' : 'Run Categorical Encoding'}
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
        <div className="my-4 p-3 bg-green-100 border border-green-400 text-green-700 rounded">
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

export default CategoricalTransformationPage;
