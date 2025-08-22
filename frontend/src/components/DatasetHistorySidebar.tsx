import React from 'react';
import { FaChevronLeft, FaChevronRight, FaHistory, FaTrash, FaEdit, FaCheck, FaTimes } from 'react-icons/fa';
import { useAuth } from '@/context/AuthContext';

// Fonction utilitaire pour afficher des notifications
const showNotification = (message: string, type: 'success' | 'error' = 'success') => {
  const notification = document.createElement('div');
  notification.className = `fixed top-4 right-4 p-4 rounded-md text-white ${
    type === 'success' ? 'bg-green-500' : 'bg-red-500'
  } shadow-lg z-50`;
  notification.textContent = message;
  document.body.appendChild(notification);
  
  // Supprimer la notification après 3 secondes
  setTimeout(() => {
    notification.remove();
  }, 3000);
};

interface Dataset {
  id: number;
  filename: string;
  upload_date: string;
  size: string;
  rows: number;
  columns: number;
}

interface DatasetHistorySidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

const DatasetHistorySidebar: React.FC<DatasetHistorySidebarProps> = ({ isOpen, onToggle }) => {
  const { token } = useAuth();
  const [datasets, setDatasets] = React.useState<Dataset[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [editingId, setEditingId] = React.useState<number | null>(null);
  const [newName, setNewName] = React.useState('');

  React.useEffect(() => {
    const fetchDatasets = async () => {
      if (!token) return;
      
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/datasets`, {
          headers: {
            'Accept': 'application/json',
            'Authorization': `Bearer ${token}`
          }
        });

        if (!response.ok) {
          throw new Error('Erreur lors du chargement des datasets');
        }

        const data = await response.json();
        setDatasets(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Une erreur est survenue');
      } finally {
        setLoading(false);
      }
    };

    fetchDatasets();
  }, [token]);

  const formatDate = (dateString: string) => {
    const options: Intl.DateTimeFormatOptions = { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    };
    return new Date(dateString).toLocaleDateString('fr-FR', options);
  };

  const handleDelete = async (id: number, filename: string) => {
    if (!window.confirm(`Êtes-vous sûr de vouloir supprimer le dataset "${filename}" ?`)) {
      return;
    }

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/datasets/${id}`, {
        method: 'DELETE',
        headers: {
          'Accept': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error('Erreur lors de la suppression du dataset');
      }

      // Mettre à jour la liste des datasets
      setDatasets(datasets.filter(dataset => dataset.id !== id));
      showNotification('Dataset supprimé avec succès', 'success');
    } catch (err) {
      console.error('Erreur lors de la suppression:', err);
      showNotification(err instanceof Error ? err.message : 'Une erreur est survenue', 'error');
    }
  };

  const startEditing = (id: number, currentName: string) => {
    setEditingId(id);
    setNewName(currentName);
  };

  const cancelEditing = () => {
    setEditingId(null);
    setNewName('');
  };

  const handleRename = async (id: number) => {
    if (!newName.trim()) {
      showNotification('Le nom ne peut pas être vide', 'error');
      return;
    }

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/datasets/${id}/rename`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ new_name: newName })
      });

      if (!response.ok) {
        throw new Error('Erreur lors du renommage du dataset');
      }

      // Mettre à jour la liste des datasets
      setDatasets(datasets.map(dataset => 
        dataset.id === id ? { ...dataset, filename: newName } : dataset
      ));
      
      setEditingId(null);
      setNewName('');
      showNotification('Dataset renommé avec succès', 'success');
    } catch (err) {
      console.error('Erreur lors du renommage:', err);
      showNotification(err instanceof Error ? err.message : 'Une erreur est survenue', 'error');
    }
  };

  return (
    <div className="h-full bg-white dark:bg-gray-800 shadow-lg overflow-y-auto border-l border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between p-4 border-b dark:border-gray-700">
        <div className="flex items-center space-x-2">
          <FaHistory className="text-blue-500" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Historique des Datasets</h3>
        </div>
        <button
          onClick={onToggle}
          className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
          aria-label={isOpen ? 'Fermer la barre latérale' : 'Ouvrir la barre latérale'}
        >
          {isOpen ? <FaChevronRight /> : <FaChevronLeft />}
        </button>
      </div>

      <div className="p-4 h-full flex flex-col">
        {loading ? (
          <div className="flex justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        ) : error ? (
          <div className="text-red-500 dark:text-red-300 text-sm p-2 bg-red-50 dark:bg-red-900/30 rounded">
            {error}
          </div>
        ) : datasets.length === 0 ? (
          <div className="text-gray-500 dark:text-gray-400 text-center py-8">
            Aucun dataset trouvé
          </div>
        ) : (
          <div className="space-y-4 flex-1 overflow-y-auto">
            {datasets.map((dataset) => (
              <div 
                key={dataset.id}
                className="p-3 border dark:border-gray-700 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors cursor-pointer"
              >
                <div className="flex justify-between items-start">
                  {editingId === dataset.id ? (
                    <div className="flex-1 flex space-x-2">
                      <input
                        type="text"
                        value={newName}
                        onChange={(e) => setNewName(e.target.value)}
                        className="flex-1 px-2 py-1 border dark:border-gray-700 rounded text-sm bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100"
                        autoFocus
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleRename(dataset.id);
                          if (e.key === 'Escape') cancelEditing();
                        }}
                      />
                      <button
                        onClick={() => handleRename(dataset.id)}
                        className="text-green-500 hover:text-green-700 p-1"
                        title="Valider"
                      >
                        <FaCheck size={14} />
                      </button>
                      <button
                        onClick={cancelEditing}
                        className="text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 p-1"
                        title="Annuler"
                      >
                        <FaTimes size={14} />
                      </button>
                    </div>
                  ) : (
                    <div 
                      className="font-medium text-gray-900 dark:text-gray-100 truncate flex-1 cursor-pointer"
                      onClick={() => console.log('Charger le dataset:', dataset.id)}
                    >
                      {dataset.filename}
                    </div>
                  )}
                  
                  <div className="flex space-x-1 ml-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        startEditing(dataset.id, dataset.filename);
                      }}
                      className="text-blue-500 hover:text-blue-700 p-1"
                      title="Renommer"
                    >
                      <FaEdit size={14} />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(dataset.id, dataset.filename);
                      }}
                      className="text-red-500 hover:text-red-700 p-1"
                      title="Supprimer"
                    >
                      <FaTrash size={14} />
                    </button>
                  </div>
                </div>
                
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                  <span>{dataset.rows} lignes × {dataset.columns} colonnes</span>
                  <span>{dataset.size}</span>
                </div>
                <div className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                  {formatDate(dataset.upload_date)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default DatasetHistorySidebar;
