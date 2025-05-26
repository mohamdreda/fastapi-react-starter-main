import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { FaSpinner } from 'react-icons/fa';
import { FileUpload } from './FileUpload';

interface Dataset {
  id: number;
  filename: string;
  file_type: string;
  created_at: string;
}

export const DatasetList: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const fetchDatasets = async () => {
    try {
      const response = await fetch('/api/v1/datasets/');
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
    fetchDatasets();
  }, []);

  const handleUploadComplete = async () => {
    setUploading(false);
    await fetchDatasets();
  };

  const handleUploadStart = () => {
    setUploading(true);
    setError(null);
  };

  const handleUploadError = (errorMessage: string) => {
    setError(errorMessage);
    setUploading(false);
  };

  const handleDelete = async (datasetId: number) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) {
      return;
    }

    try {
      const response = await fetch(`/api/v1/datasets/${datasetId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to delete dataset');
      }

      setDatasets(datasets.filter(d => d.id !== datasetId));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete dataset');
    }
  };

  if (loading || uploading) {
    return (
      <div className="flex justify-center p-8">
        <FaSpinner className="w-8 h-8 animate-spin text-primary-500" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg p-6">
        <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Upload New Dataset</h2>
        <FileUpload
          onUploadComplete={handleUploadComplete}
          onUploadStart={handleUploadStart}
          onUploadError={handleUploadError}
        />
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 p-4 rounded-md">
          {error}
        </div>
      )}

      <div className="bg-white dark:bg-gray-800 shadow dark:shadow-sm rounded-lg overflow-hidden">
        <div className="p-6 text-gray-900 dark:text-white">
          <h2 className="text-lg font-semibold mb-4">Datasets</h2>
          {datasets.length > 0 ? (
            <div className="overflow-x-auto text-gray-900 dark:text-white">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-700">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Uploaded
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                  {datasets.map((dataset) => (
                    <tr key={dataset.id}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                        {dataset.filename}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {dataset.file_type}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(dataset.created_at).toLocaleDateString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <div className="flex items-center space-x-2">
                          <Link
                            to={`/dashboard/diagnosis/${dataset.id}`}
                            className="inline-flex items-center px-2 py-1 text-sm font-medium text-white bg-primary-600 rounded-md hover:bg-primary-700"
                            title="Analyze Dataset"
                          >
                            <FaSpinner className="w-4 h-4 mr-1" />
                            Analyze
                          </Link>
                          <button
                            onClick={() => handleDelete(dataset.id)}
                            className="inline-flex items-center px-2 py-1 text-sm font-medium text-white bg-red-600 rounded-md hover:bg-red-700"
                            title="Delete Dataset"
                          >
                            <FaSpinner className="w-4 h-4 mr-1" />
                            Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-gray-500 dark:text-gray-400 text-center py-4">
              No datasets uploaded yet.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};
