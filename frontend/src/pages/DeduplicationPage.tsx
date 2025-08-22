import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';

// Interfaces pour une meilleure typage
interface Dataset {
  id: number;
  filename: string;
}

interface ParamsState {
  fuzzy: { 
    threshold: number; 
    method: string; 
    case_sensitive: boolean;
    include_numeric: boolean;
    max_matches: number;
    columns: string[];
  };
  deep: { threshold: number; batch_size: number };
}

const similarityMethods = [
  { label: 'Standard Ratio', value: 'ratio' },
  { label: 'Partial Ratio', value: 'partial_ratio' },
  { label: 'Token Sort Ratio', value: 'token_sort_ratio' },
  { label: 'Token Set Ratio', value: 'token_set_ratio' },
];

const algorithms = [
  { key: 'fuzzy', label: 'Fuzzy Matching' },
  { key: 'deep', label: 'Deep ER' },
];

const DeduplicationPage: React.FC = () => {
  const { token } = useAuth();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedAlgo, setSelectedAlgo] = useState<keyof ParamsState>('fuzzy'); // Type plus strict
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [removeDuplicates, setRemoveDuplicates] = useState<boolean>(false);
  const [outputFilename, setOutputFilename] = useState<string>('');
  
  // États pour le résultat de la déduplication
  const [dedupLoading, setDedupLoading] = useState<boolean>(false);
  const [dedupError, setDedupError] = useState<string | null>(null);
  const [dedupResult, setDedupResult] = useState<any>(null);

  // Paramètres par défaut pour chaque algorithme
  const [params, setParams] = useState<ParamsState>({
    fuzzy: { 
      threshold: 0.8, 
      method: 'ratio',
      case_sensitive: false,
      include_numeric: true,
      max_matches: 100,
      columns: []
    },

    deep: { threshold: 0.8, batch_size: 32 },
  });

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

  const handleParamChange = (algo: keyof ParamsState, param: string, value: any) => {
    setParams(prev => ({
      ...prev,
      [algo]: {
        ...prev[algo],
        [param]: value,
      },
    }));
  };

  const handleRunDeduplication = async () => {
    if (!selectedDataset) return;

    setDedupLoading(true);
    setDedupError(null);
    setDedupResult(null);

    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/deduplication`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          dataset_id: Number(selectedDataset),
          algorithm: selectedAlgo,
          params: params[selectedAlgo],
          remove_duplicates: removeDuplicates,
          output_filename: outputFilename.trim() || null
        })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Deduplication failed');
      setDedupResult(data);
    } catch (e: any) {
      setDedupError(e.message);
    } finally {
      setDedupLoading(false);
    }
  };
  
  // Function to handle downloading the cleaned dataset
  const handleDownloadFile = async (filePath: string) => {
    try {
      // Normalize the file path for both Windows and Unix systems
      // Replace backslashes with forward slashes for consistent handling
      const normalizedPath = filePath.replace(/\\/g, '/');
      
      console.log('Downloading file:', normalizedPath);
      
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/datasets/file/download?path=${encodeURIComponent(normalizedPath)}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Download error response:', response.status, errorText);
        throw new Error(`Failed to download file: ${response.status} ${errorText}`);
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      
      // Extract filename from path - handle both slash types
      const filename = normalizedPath.split('/').pop() || 
                      normalizedPath.split('\\').pop() || 
                      'cleaned_dataset.csv';
      a.download = filename;
      
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error downloading file:', error);
      setDedupError('Failed to download the cleaned dataset');
    }
  };

  const renderAlgoParams = () => {
    switch (selectedAlgo) {
      case 'fuzzy':
        return (
          <div className="space-y-4">
            <div>
              <label htmlFor="threshold" className="block text-sm font-medium text-gray-700 mb-1">
                Similarity Threshold: <span className="font-bold">{params.fuzzy.threshold}</span>
              </label>
              <input 
                id="threshold" 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                value={params.fuzzy.threshold} 
                onChange={(e) => handleParamChange('fuzzy', 'threshold', parseFloat(e.target.value))} 
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            <div>
              <label htmlFor="sim-method" className="block text-sm font-medium text-gray-700 mb-1">Similarity Method</label>
              <select 
                id="sim-method" 
                value={params.fuzzy.method} 
                onChange={(e) => handleParamChange('fuzzy', 'method', e.target.value)} 
                className="w-full p-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
              >
                {similarityMethods.map(opt => (<option key={opt.value} value={opt.value}>{opt.label}</option>))}
              </select>
            </div>
            
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="case-sensitive"
                checked={params.fuzzy.case_sensitive}
                onChange={(e) => handleParamChange('fuzzy', 'case_sensitive', e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="case-sensitive" className="text-sm text-gray-700">
                Case Sensitive Matching
              </label>
            </div>
            
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="include-numeric"
                checked={params.fuzzy.include_numeric}
                onChange={(e) => handleParamChange('fuzzy', 'include_numeric', e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="include-numeric" className="text-sm text-gray-700">
                Include Numeric Fields
              </label>
            </div>
            
            <div>
              <label htmlFor="max-matches" className="block text-sm font-medium text-gray-700 mb-1">
                Maximum Matches: <span className="font-bold">{params.fuzzy.max_matches}</span>
              </label>
              <input 
                id="max-matches" 
                type="range" 
                min="10" 
                max="1000" 
                step="10" 
                value={params.fuzzy.max_matches} 
                onChange={(e) => handleParamChange('fuzzy', 'max_matches', parseInt(e.target.value))} 
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
        );
      

      case 'deep':
        return (
          <div className="space-y-4">
            <div>
              <label htmlFor="threshold" className="block text-sm font-medium text-gray-700 mb-1">
                Similarity Threshold: <span className="font-bold">{params.deep.threshold}</span>
              </label>
              <input id="threshold" type="range" min="0" max="1" step="0.01" value={params.deep.threshold} onChange={(e) => handleParamChange('deep', 'threshold', parseFloat(e.target.value))} className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"/>
            </div>
            <div>
              <label htmlFor="batch-size" className="block text-sm font-medium text-gray-700 mb-1">Batch Size</label>
              <input id="batch-size" type="number" min="1" value={params.deep.batch_size} onChange={(e) => handleParamChange('deep', 'batch_size', parseInt(e.target.value))} className="w-full p-2 border rounded-md"/>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  // CORRECTION : La structure JSX est maintenant propre et sans duplication.
  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6">Duplicates Detection</h1>

      {/* Carte de sélection du Dataset */}
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

      {/* Carte des algorithmes et paramètres */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold mb-4">Algorithm & Parameters</h2>
        
        {/* Options for duplicate removal and output filename */}
        <div className="mb-6 space-y-4">
          <div className="flex items-center">
            <input
              type="checkbox"
              id="remove-duplicates"
              checked={removeDuplicates}
              onChange={(e) => setRemoveDuplicates(e.target.checked)}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <label htmlFor="remove-duplicates" className="ml-2 block text-sm text-gray-900">
              Remove duplicates from dataset
            </label>
          </div>
          
          {removeDuplicates && (
            <div className="pl-6 border-l-2 border-blue-100">
              <label htmlFor="output-filename" className="block text-sm font-medium text-gray-700 mb-1">
                Output Filename (optional)
              </label>
              <input
                type="text"
                id="output-filename"
                value={outputFilename}
                onChange={(e) => setOutputFilename(e.target.value)}
                placeholder="e.g., cleaned_dataset (extension will be added automatically)"
                className="w-full p-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
              />
              <p className="mt-1 text-xs text-gray-500">
                Leave empty to use default naming convention
              </p>
            </div>
          )}
        </div>
        
        {/* Onglets des algorithmes */}
        <div className="flex border-b mb-6">
          {algorithms.map(algo => (
            <button
              key={algo.key}
              onClick={() => setSelectedAlgo(algo.key as keyof ParamsState)}
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

        {/* Paramètres de l'algorithme sélectionné */}
        <div>
          {renderAlgoParams()}
        </div>

        {/* Bouton d'action */}
        <div className="mt-8">
          <button
            type="button"
            className="w-full bg-blue-600 text-white font-bold py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400"
            disabled={!selectedDataset || dedupLoading}
            onClick={handleRunDeduplication}
          >
            {dedupLoading ? 'Running...' : 'Run Deduplication'}
          </button>
        </div>
      </div>

      {/* Affichage du résultat ou erreur */}
      <div className="mt-4 min-h-[32px]">
        {dedupError && (
          <div className="text-red-500 bg-red-50 p-3 rounded-md">{dedupError}</div>
        )}
        {dedupResult && (
          <div className="text-green-600 bg-green-50 p-3 rounded-md">
            <div className="font-semibold">{dedupResult.status === 'success' ? 'Success!' : 'Result'}</div>
            <div>{dedupResult.message}</div>
            {dedupResult.num_duplicates !== undefined && (
              <div>Duplicates found: {dedupResult.num_duplicates}</div>
            )}
            {dedupResult.duplicates_removed && (
              <div className="mt-2 font-medium text-blue-600">
                ✓ Duplicates were successfully removed from the dataset
                {dedupResult.cleaned_dataset_path && (
                  <div className="text-sm text-gray-600 mt-1">
                    Cleaned dataset saved to: {dedupResult.cleaned_dataset_path.split('/').pop()}
                    <button
                      onClick={() => handleDownloadFile(dedupResult.cleaned_dataset_path)}
                      className="mt-2 inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                    >
                      <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                      </svg>
                      Download Cleaned Dataset
                    </button>
                  </div>
                )}
              </div>
            )}
            {/* Model Information for Random Forest */}
            {selectedAlgo === 'rf' && dedupResult.model_info && (
              <div className="mt-4 bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                <h3 className="text-md font-semibold text-gray-800 mb-2">Model Information</h3>
                
                {/* Feature Importance */}
                <div className="mb-3">
                  <h4 className="text-sm font-medium text-gray-700">Top Important Features:</h4>
                  <div className="text-sm text-gray-600">{dedupResult.model_info.top_features}</div>
                </div>
                
                {/* Parameters Used */}
                <div>
                  <h4 className="text-sm font-medium text-gray-700">Parameters Used:</h4>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="text-gray-600">Trees: <span className="font-medium">{dedupResult.model_info.parameters.n_estimators}</span></div>
                    <div className="text-gray-600">Max Depth: <span className="font-medium">{dedupResult.model_info.parameters.max_depth}</span></div>
                    <div className="text-gray-600">Class Weight: <span className="font-medium">{dedupResult.model_info.parameters.class_weight || 'None'}</span></div>
                    <div className="text-gray-600">Min Samples Leaf: <span className="font-medium">{dedupResult.model_info.parameters.min_samples_leaf}</span></div>
                    <div className="text-gray-600">Features Per Split: <span className="font-medium">{dedupResult.model_info.parameters.max_features}</span></div>
                    <div className="text-gray-600">Criterion: <span className="font-medium">{dedupResult.model_info.parameters.criterion}</span></div>
                    <div className="text-gray-600">Confidence Threshold: <span className="font-medium">{dedupResult.model_info.parameters.threshold}</span></div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Result Preview */}
            {dedupResult.result_preview && (
              <div className="mt-4">
                <h3 className="text-md font-semibold text-gray-800 mb-2">Duplicate Pairs Preview</h3>
                <pre className="bg-gray-100 rounded p-2 text-xs overflow-x-auto">{JSON.stringify(dedupResult.result_preview, null, 2)}</pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DeduplicationPage;