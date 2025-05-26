import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { FaCloudUploadAlt } from 'react-icons/fa';
import { useAuth } from '@/context/AuthContext';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface FileUploadProps {
  onUploadComplete: () => void;
  onUploadStart: () => void;
  onUploadError: (error: string) => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onUploadComplete,
  onUploadStart,
  onUploadError,
}) => {
  const [uploading, setUploading] = useState(false);
  const { token } = useAuth();

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    const fileType = file.name.split('.').pop()?.toLowerCase();

    if (!fileType || !['csv', 'xlsx', 'xls', 'json', 'parquet', 'feather', 'tsv'].includes(fileType)) {
      onUploadError('Unsupported file format');
      return;
    }

    const formData = new FormData();
    // Utiliser le nom original du fichier explicitement
    formData.append('file', file, file.name);
    formData.append('file_type', fileType);
    formData.append('original_filename', file.name);

    setUploading(true);
    onUploadStart();

    try {
      const response = await fetch(`${API_URL}/api/v1/upload`, {
        method: 'POST',
        body: formData,
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
      }

      onUploadComplete();
    } catch (error) {
      onUploadError(error instanceof Error ? error.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, [onUploadComplete, onUploadStart, onUploadError, token]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/json': ['.json'],
      'application/octet-stream': ['.parquet', '.feather'],
      'text/tab-separated-values': ['.tsv']
    },
    maxSize: 500 * 1024 * 1024, // 500MB
    multiple: false
  });

  return (
    <div
      {...getRootProps()}
      className={`
        border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
        transition-colors duration-200 ease-in-out
        ${isDragActive ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20' : 'border-gray-300 dark:border-gray-700 hover:border-primary-500'}
        ${uploading ? 'opacity-50 cursor-not-a
