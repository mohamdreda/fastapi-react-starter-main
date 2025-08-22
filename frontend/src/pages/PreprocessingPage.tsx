import React, { useEffect, useState } from 'react';
import { DownloadIcon } from 'lucide-react';

import { useAuth } from '../context/AuthContext';
import { useSearchParams } from 'react-router-dom';

interface Dataset {
  id: number;
  filename: string;
}

const PreprocessingPage: React.FC = () => {
  const { token } = useAuth();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [columns, setColumns] = useState<string[]>([]);
  const [loadingColumns, setLoadingColumns] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [textFields, setTextFields] = useState<string[]>([]);
  const [numericFields, setNumericFields] = useState<string[]>([]);
  const [categoricalFields, setCategoricalFields] = useState<string[]>([]);

  const [summary, setSummary] = useState<any | null>(null);
  const [preprocessedPath, setPreprocessedPath] = useState<string | null>(null);
  const [downloadName, setDownloadName] = useState<string>('');
  const [sessionStepId, setSessionStepId] = useState<string | null>(null);

  const [searchParams] = useSearchParams();
  const urlSessionId = searchParams.get('session_id');
  const storedSessionId = typeof window !== 'undefined' ? localStorage.getItem('active_session_id') : null;
  const sessionId = urlSessionId || storedSessionId || null;

  const handleDownload = async () => {
    if (!preprocessedPath || !token) return;
    try {
      const url = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${preprocessedPath.replace(/^[.]/, '')}`;
      const response = await fetch(url, { headers: { 'Authorization': `Bearer ${token}` } });
      if (!response.ok) {
        throw new Error('Erreur lors du téléchargement du fichier');
      }
      const blob = await response.blob();
      const filename = `${downloadName || 'preprocessed_data'}.csv`;
      const urlBlob = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = urlBlob;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(urlBlob);
    } catch (err) {
      console.error(err);
    }
  };

  // Fetch datasets on mount
  useEffect(() => {
    if (!token) return;
    fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/datasets`, {
      headers: {
        'Accept': 'application/json',
        'Authorization': `Bearer ${token}`
      }
    })
      .then(async res => {
        if (!res.ok) throw new Error('Erreur lors du chargement des datasets');
        const data = await res.json();
        setDatasets(data);
      })
      .catch(() => setDatasets([]));
  }, [token]);

  // Fetch columns when a dataset is selected
  useEffect(() => {
    if (!selectedDataset || !token) {
      setColumns([]);
      setTextFields([]);
      setNumericFields([]);
      setCategoricalFields([]);
      setErrorMsg(null);
      return;
    }

    const fetchColumns = async () => {
      setLoadingColumns(true);
      setErrorMsg(null);
      
      try {
        const response = await fetch(
          `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/datasets/${selectedDataset}/columns`,
          {
            headers: {
              'Accept': 'application/json',
              'Authorization': `Bearer ${token}`
            }
          }
        );

        if (!response.ok) {
          let errorMessage = 'Failed to load columns';
          try {
            const errorData = await response.json();
            errorMessage = errorData.detail || errorMessage;
          } catch (e) {
            // If we can't parse the error as JSON, use the status text
            errorMessage = response.statusText || errorMessage;
          }
          throw new Error(errorMessage);
        }

        const columns = await response.json();
        
        if (!Array.isArray(columns)) {
          throw new Error('Invalid response format: expected array of columns');
        }

        setColumns(columns);
        setTextFields(columns.filter(col => 
          typeof col === 'string' && 
          ['name', 'title', 'description', 'text', 'comment'].some(term => 
            col.toLowerCase().includes(term)
          )
        ));
        
        setNumericFields(columns.filter(col => 
          typeof col === 'string' && 
          ['id', 'price', 'amount', 'quantity', 'count', 'total', 'number'].some(term => 
            col.toLowerCase().includes(term)
          )
        ));
        
        setCategoricalFields(columns.filter(col => 
          typeof col === 'string' && 
          ['type', 'category', 'status', 'gender', 'color', 'size'].some(term => 
            col.toLowerCase().includes(term)
          )
        ));
        
        if (columns.length === 0) {
          setErrorMsg('This dataset has no columns.');
        }
      } catch (error) {
        console.error('Error fetching columns:', error);
        setErrorMsg(error instanceof Error ? error.message : 'Failed to load columns');
        setColumns([]);
      } finally {
        setLoadingColumns(false);
      }
    };

    fetchColumns();
  }, [selectedDataset, token]);

  const handleMultiSelect = (setter: React.Dispatch<React.SetStateAction<string[]>>, values: string[]) => {
    setter(values);
  };

  const handlePreprocessing = async () => {
    if (!selectedDataset || !token) return;
    
    setErrorMsg(null);
    setSessionStepId(null);

    try {
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const sessionQS = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
      const response = await fetch(
        `${API_URL}/api/v1/deduplication/pipeline/preprocessing${sessionQS}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({
            dataset_id: Number(selectedDataset),
            text_columns: textFields,
            numeric_columns: numericFields,
            categorical_columns: categoricalFields
          })
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Error during preprocessing');
      }

      const resultData = await response.json();
      if (typeof resultData.session_step_id !== 'undefined') {
        setSessionStepId(resultData.session_step_id || null);
      }
      if (resultData.status !== 'success') {
        throw new Error(resultData.message || 'Error during preprocessing');
      }
      
      setSummary(resultData.summary || null);
      setPreprocessedPath(resultData.preprocessed_data_path || null);
    } catch (error) {
      console.error('Preprocessing error:', error);
      setErrorMsg(error instanceof Error ? error.message : 'Failed to preprocess data');
    }
  };

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-2 text-center text-gray-800">Préparation et Prétraitement des Données</h1>
      <div className="text-sm text-gray-600 text-right mb-6">
        {sessionId && (
          <span>
            Session active: <span className="font-mono">{sessionId}</span>
          </span>
        )}
        {sessionStepId && (
          <span>
            {sessionId ? ' • ' : ''}Dernière étape: <span className="font-mono">{sessionStepId}</span>
          </span>
        )}
      </div>
      {/* Dataset selection */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <h2 className="text-lg font-semibold mb-3">Prétraitement</h2>

        <h2 className="text-lg font-semibold mb-3">Sélection du Dataset</h2>
        <select
          value={selectedDataset}
          onChange={e => setSelectedDataset(e.target.value)}
          className="w-full p-2 border rounded"
          disabled={!datasets.length}
        >
          <option value="">Sélectionner un dataset</option>
          {datasets.map(ds => (
            <option key={ds.id} value={ds.id}>{ds.filename}</option>
          ))}
        </select>
        {selectedDataset && (
          <div className="mt-2 text-sm text-gray-600">
            <span className="font-semibold">Dataset sélectionné :</span> {datasets.find(ds => ds.id.toString() === selectedDataset)?.filename}
          </div>
        )}
      </div>

      {/* Configuration bloc */}
      {selectedDataset && (
        <>
          <div className="mb-6 bg-white p-4 rounded-lg shadow">
            <h2 className="text-lg font-semibold mb-3">Configuration du Prétraitement</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Champs Texte</label>
                <select
                  multiple
                  value={textFields}
                  onChange={e => handleMultiSelect(setTextFields, Array.from(e.target.selectedOptions, opt => opt.value))}
                  className="w-full p-2 border rounded h-28"
                  disabled={loadingColumns || columns.length === 0}
                  size={Math.min(8, columns.length || 1)}
                >
                  {columns.map((col: string) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Champs Numériques</label>
                <select
                  multiple
                  value={numericFields}
                  onChange={e => handleMultiSelect(setNumericFields, Array.from(e.target.selectedOptions, opt => opt.value))}
                  className="w-full p-2 border rounded h-28"
                  disabled={loadingColumns || columns.length === 0}
                  size={Math.min(8, columns.length || 1)}
                >
                  {columns.map((col: string) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Champs Catégoriels</label>
                <select
                  multiple
                  value={categoricalFields}
                  onChange={e => handleMultiSelect(setCategoricalFields, Array.from(e.target.selectedOptions, opt => opt.value))}
                  className="w-full p-2 border rounded h-28"
                  disabled={loadingColumns || columns.length === 0}
                  size={Math.min(8, columns.length || 1)}
                >
                  {columns.map((col: string) => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
              </div>
            </div>
            {loadingColumns && (
              <div className="text-blue-600 text-sm mt-2">Chargement des colonnes...</div>
            )}
            {errorMsg && (
              <div className="text-red-600 text-sm mt-2">{errorMsg}</div>
            )}
            <div className="mt-4">
              <h3 className="text-md font-semibold text-gray-700 mb-2">Options de Prétraitement</h3>
              <ul className="list-disc ml-6 text-gray-700">
                <li><b>Texte</b> : Nettoyage basique (minuscule, strip)</li>
                <li><b>Numérique</b> : Min-Max Scaling</li>
                <li><b>Catégoriel</b> : Encodage par labels</li>
              </ul>
            </div>
          </div>

          {/* Actions */}
          <div className="mb-6 bg-white p-4 rounded-lg shadow flex flex-wrap gap-4">
            <button
              className="px-4 py-2 rounded text-white bg-blue-500 hover:bg-blue-600 font-semibold"
              onClick={handlePreprocessing}
              disabled={loadingColumns || columns.length === 0}
            >
              Lancer le prétraitement
            </button>
          </div>

          {/* Summary and download */}
          {summary && (
            <div className="my-6 bg-white p-4 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-3">Résumé du Prétraitement</h3>
              {preprocessedPath && (
                <div className="mb-4 flex items-center gap-2">
                  <input
                    type="text"
                    className="flex-1 p-2 border rounded"
                    placeholder="Nom de fichier personnalisé (optionnel)"
                    value={downloadName}
                    onChange={e => setDownloadName(e.target.value)}
                  />
                  <button
                    onClick={handleDownload}
                    className="px-4 py-2 rounded text-white bg-gray-600 hover:bg-gray-700 font-semibold flex items-center gap-2"
                  >
                    <DownloadIcon size={16}/> Télécharger les données prétraitées
                  </button>
                </div>
              )}
              <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
                <li><span className="font-medium">Lignes traitées :</span> {summary.rows_processed}</li>
                <li><span className="font-medium">Colonnes finales :</span> {summary.columns_processed}</li>
                <li><span className="font-medium">Texte nettoyé :</span> {summary.text_columns_cleaned?.join(', ') || 'Aucun'}</li>
                <li><span className="font-medium">Numérique mis à l'échelle :</span> {summary.numeric_scaled_columns?.join(', ') || 'Aucun'}</li>
                <li><span className="font-medium">Catégoriel encodé :</span> {summary.categorical_encoded_columns?.join(', ') || 'Aucun'}</li>
              </ul>
            </div>
          )}
        </>
      )}


    </div>
  );
};

export default PreprocessingPage;
