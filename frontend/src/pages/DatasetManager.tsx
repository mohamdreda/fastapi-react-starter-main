import React, { useState, useEffect } from 'react';
import { FileUpload } from '@/components/FileUpload';
import {
  FaTable,
  FaTrash,
  FaChartBar,
  FaExclamationTriangle,
  FaCheck,
  FaSpinner,
} from 'react-icons/fa';
import { useNavigate, useParams } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import ConfirmationDialog from '@/components/ui/ConfirmationDialog';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface Dataset {
  id: number;
  filename: string;
  file_type: string;
  format: string;
  missing_values: {
    total_missing: number;
    missing_percentage: number;
    per_column: Record<string, { count: number; percentage: number }>;
  };
  duplicates: number;
  data_types: Record<string, string>;
  categorical_issues: Record<string, { inconsistent_format?: string[]; rare_values?: string[] }>;
  summary_stats: Record<string, Record<string, number>>;
  created_at: string;
}

export const DatasetManager: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const { userId } = useParams();
  const navigate = useNavigate();
  const { token } = useAuth();

  const fetchDatasets = async () => {
    try {
      const response = await fetch(`${API_URL}/api/v1/datasets`, {
        headers: {
          'Accept': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch datasets');
      }

      const data = await response.json();
      setDatasets(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load datasets');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (token) {
      fetchDatasets();
    }
  }, [token]);

  const handleUploadComplete = async () => {
    setUploading(false);
    await fetchDatasets(); // Refresh list after upload
  };

  const handleUploadStart = () => {
    setUploading(true);
    setError(null);
  };

  const handleUploadError = (errorMessage: string) => {
    setError(errorMessage);
    setUploading(false);
  };

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);

  const handleDelete = async (datasetId: number) => {
    setSelectedDatasetId(datasetId);
    setDeleteDialogOpen(true);
  };

  const confirmDelete = async () => {
    if (!selectedDatasetId) return;

    try {
      const response = await fetch(`${API_URL}/api/v1/datasets/${selectedDatasetId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to delete dataset');
      }

      setDatasets(datasets.filter(d => d.id !== selectedDatasetId));
      setDeleteDialogOpen(false);
      setSelectedDatasetId(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete dataset');
      setDeleteDialogOpen(false);
      setSelectedDatasetId(null);
    }
  };

  const getDataQualityIndicator = (dataset: Dataset) => {
    const missingPercentage = dataset.missing_values?.missing_percentage || 0;
    const duplicatePercentage = (dataset.duplicates || 0) * 100;
    const totalIssues = missingPercentage + duplicatePercentage;

    if (totalIssues < 10) return <FaCheck className="text-green-500" />;
    if (totalIssues < 30) return <FaExclamationTriangle className="text-yellow-500" />;
    return <FaExclamationTriangle className="text-red-500" />;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <FaSpinner className="animate-spin text-4xl text-blue-500" />
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold mb-8">Dataset Manager</h1>

      <div className="bg-white rounded-lg shadow p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Upload New Dataset</h2>
        <FileUpload
          onUploadComplete={handleUploadComplete}
          onUploadStart={handleUploadStart}
          onUploadError={handleUploadError}
        />
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Your Datasets</h2>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        {datasets.length === 0 ? (
          <div className="text-center py-8">
            <div className="flex justify-center mb-4">
              <FaTable className="text-4xl text-gray-400" />
            </div>
            <p className="text-gray-500">No datasets</p>
            <p className="text-gray-400 text-sm">Get started by uploading your first dataset</p>
          </div>
        ) : (
          <div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
            {datasets.map((dataset) => (
              <div key={dataset.id} className="border rounded-lg p-4">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="font-semibold">{dataset.filename}</h3>
                    <p className="text-sm text-gray-500">{dataset.file_type}</p>
                  </div>
                  {getDataQualityIndicator(dataset)}
                </div>

                <div className="text-sm text-gray-600 mb-4">
                  <p>Missing Values: {(dataset.missing_values?.missing_percentage ?? 0).toFixed(1)}%</p>
                  <p>Duplicates: {dataset.duplicates || 0}</p>
                  <p>Created: {new Date(dataset.created_at).toLocaleDateString()}</p>
                </div>

                <div className="flex justify-between items-center">
                  <button
                    onClick={() => navigate(`/user/dashboard/${userId}/diagnosis/${dataset.id}`)}
                    className="text-blue-500 hover:text-blue-700"
                  >
                    <FaChartBar className="inline mr-1" /> Analyze
                  </button>
                  <button
                    onClick={() => handleDelete(dataset.id)}
                    className="text-red-500 hover:text-red-700"
                  >
                    <FaTrash className="inline mr-1" /> Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      <ConfirmationDialog
        open={deleteDialogOpen}
        onClose={() => {
          setDeleteDialogOpen(false);
          setSelectedDatasetId(null);
        }}
        onConfirm={confirmDelete}
      />
    </div>
  );
};

export default DatasetManager;
