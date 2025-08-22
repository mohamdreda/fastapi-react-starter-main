import React, { useState, useEffect, useRef } from 'react';
// OutputSidebar removed
import DatasetHistorySidebar from './DatasetHistorySidebar';
import { Link, useLocation, useParams, useNavigate } from 'react-router-dom';
import { 
  FaHistory, 
  FaTable, 
  FaSearch, 
  FaExclamationTriangle, 
  FaRandom, 
  FaChartBar, 
  FaDownload, 
  FaCog, 
  FaMagic, 
  FaSortNumericDown, 
  FaRuler, 
  FaRobot, 
  FaHandSparkles, 
  FaMicroscope, 
  FaSitemap, 
  FaNetworkWired, 
  FaTree, 
  FaTh,
  FaProjectDiagram 
} from 'react-icons/fa';
import Logo from '@/assets/data-cleaning-logo.svg';
import { ThemeToggle } from './ui/ThemeToggle';
import { useAuth } from '../context/AuthContext';

interface Tool {
  id: string;
  name: string;
  icon: React.ReactNode;
  path: string;
  description: string;
  subTools?: Tool[];
}

interface GalaxyLayoutProps {
  children: React.ReactNode;
}

const createTools = (userId: string): Tool[] => {
  return [
    {
      id: 'datasets',
      name: 'Datasets',
      icon: <FaTable className="text-2xl" />,
      path: `/user/dashboard/${userId}/datasets`,
      description: 'Manage your uploaded datasets'
    },
    {
      id: 'diagnosis',
      name: 'Data Diagnosis',
      icon: <FaSearch className="text-xl" />,
      path: `/user/dashboard/${userId}/diagnosis`,
      description: 'Analyze data quality and statistics'
    },
    {
      id: 'transformation',
      name: 'Data Transformation',
      icon: <FaMagic className="text-xl" />,
      path: `/user/dashboard/${userId}/transformation`,
      description: 'Transform and normalize data',
      subTools: [
        {
          id: 'categorical',
          name: 'Handling Categorical Data',
          icon: <FaSortNumericDown className="text-lg" />,
          path: `/user/dashboard/${userId}/transformation/categorical`,
          description: 'Convert categorical data to numerical formats'
        },
        {
          id: 'standardization',
          name: 'Standardization',
          icon: <FaRuler className="text-lg" />,
          path: `/user/dashboard/${userId}/transformation/standardization`,
          description: 'Scale and normalize numerical data'
        }
      ]
    },
    {
      id: 'feature-engineering',
      name: 'Feature Engineering',
      icon: <FaRobot className="text-xl" />,
      path: `/user/dashboard/${userId}/feature-engineering`,
      description: 'Advanced feature manipulation and selection',
      subTools: [
        {
          id: 'autoencoder',
          name: 'Autoencoder',
          icon: <FaMicroscope className="text-lg" />,
          path: `/user/dashboard/${userId}/feature-engineering/autoencoder`,
          description: 'Deep learning based feature extraction'
        },
        {
          id: 'selection',
          name: 'Feature Selection',
          icon: <FaHandSparkles className="text-lg" />,
          path: `/user/dashboard/${userId}/feature-engineering/selection`,
          description: 'Select most relevant features'
        },
        {
          id: 'extraction',
          name: 'Feature Extraction',
          icon: <FaRobot className="text-lg" />,
          path: `/user/dashboard/${userId}/feature-engineering/extraction`,
          description: 'Extract meaningful features from raw data'
        }
      ]
    },
    {
      id: 'clustering',
      name: 'Clustering',
      icon: <FaSitemap className="text-xl" />,
      path: `/user/dashboard/${userId}/clustering`,
      description: 'Group similar data points together',
      subTools: [
        {
          id: 'partitioning',
          name: 'Partitioning methods',
          icon: <FaNetworkWired className="text-lg" />,
          path: `/user/dashboard/${userId}/clustering/partitioning`,
          description: 'K-means and related algorithms'
        },
        {
          id: 'density',
          name: 'Density-based methods',
          icon: <FaTree className="text-lg" />,
          path: `/user/dashboard/${userId}/clustering/density`,
          description: 'DBSCAN and HDBSCAN'
        },
        {
          id: 'hierarchical',
          name: 'Hierarchical methods',
          icon: <FaSitemap className="text-lg" />,
          path: `/user/dashboard/${userId}/clustering/hierarchical`,
          description: 'Agglomerative and divisive clustering'
        },
        {
          id: 'grid',
          name: 'Grid-based methods',
          icon: <FaTh className="text-lg" />,
          path: `/user/dashboard/${userId}/clustering/grid`,
          description: 'STING and CLIQUE algorithms'
        }
      ]
    },
    {
      id: 'outlier-detection',
      name: 'Outlier Detection',
      icon: <FaChartBar className="text-xl" />,
      path: `/user/dashboard/${userId}/outlier-detection`,
      description: 'Detect and analyze outliers in your data',
      subTools: [
        {
          id: 'statistical',
          name: 'Statistical-based Methods',
          icon: <FaSortNumericDown className="text-lg" />,
          path: `/user/dashboard/${userId}/outlier-detection/statistical`,
          description: 'Detect outliers using statistical properties'
        },
        {
          id: 'distance',
          name: 'Distance-based Methods',
          icon: <FaRuler className="text-lg" />,
          path: `/user/dashboard/${userId}/outlier-detection/distance`,
          description: 'Outlier detection based on distance metrics'
        },
        {
          id: 'clustering',
          name: 'Clustering-based Methods',
          icon: <FaSitemap className="text-lg" />,
          path: `/user/dashboard/${userId}/outlier-detection/clustering`,
          description: 'Use clustering algorithms to find outliers'
        },
        {
          id: 'ml',
          name: 'Machine Learning-based Methods',
          icon: <FaRobot className="text-lg" />,
          path: `/user/dashboard/${userId}/outlier-detection/ml`,
          description: 'Detect outliers with ML models'
        },
        {
          id: 'density',
          name: 'Density-based Methods',
          icon: <FaTree className="text-lg" />,
          path: `/user/dashboard/${userId}/outlier-detection/density`,
          description: 'Density-based outlier detection methods'
        },
        {
          id: 'ensemble',
          name: 'Ensemble Methods',
          icon: <FaTh className="text-lg" />,
          path: `/user/dashboard/${userId}/outlier-detection/ensemble`,
          description: 'Combine multiple techniques for robust detection'
        }
      ]
    },
    {
      id: 'imputation',
      name: 'Data Imputation',
      icon: <FaExclamationTriangle className="text-xl" />,
      path: `/user/dashboard/${userId}/imputation`,
      description: 'Handle missing values in your data'
    },
    {
      id: 'duplicates',
      name: 'Data Deduplication',
      icon: <FaRandom className="text-xl" />,
      path: `/user/dashboard/${userId}/duplicates`,
      description: 'Detect and remove duplicate records',
      subTools: [
        {
          id: 'data-dedup',
          name: 'Data Dedup',
          icon: <FaRandom className="text-lg" />,
          path: `/user/dashboard/${userId}/duplicates/data-dedup`,
          description: 'Module de déduplication principale'
        },
        {
          id: 'preprocessing',
          name: 'Préparation & Prétraitement',
          icon: <FaCog className="text-lg" />,
          path: `/user/dashboard/${userId}/duplicates/preprocessing`,
          description: 'Préparation et prétraitement des données avant la déduplication'
        },
        {
          id: 'blocking',
          name: 'Blocking',
          icon: <FaTh className="text-lg" />,
          path: `/user/dashboard/${userId}/duplicates/blocking`,
          description: 'Génération de paires candidates (Blocking)'
        },
        {
          id: 'similarity',
          name: 'Calcul de Similarité',
          icon: <FaSitemap className="text-lg" />,
          path: `/user/dashboard/${userId}/duplicates/similarity`,
          description: 'Calcul de la similarité entre paires'
        },
        {
          id: 'classification',
          name: 'Classification',
          icon: <FaNetworkWired className="text-lg" />,
          path: `/user/dashboard/${userId}/duplicates/classification`,
          description: 'Classification des doublons'
        },
        {
          id: 'clustering',
          name: 'Clustering',
          icon: <FaTree className="text-lg" />,
          path: `/user/dashboard/${userId}/duplicates/clustering`,
          description: 'Regroupement des doublons détectés'
        },
        {
          id: 'results',
          name: 'Résultats & Résolution',
          icon: <FaChartBar className="text-lg" />,
          path: `/user/dashboard/${userId}/duplicates/results`,
          description: 'Résultats et résolution des conflits'
        }
      ]
    },
    {
      id: 'workflow',
      name: 'Workflow',
      icon: <FaProjectDiagram className="text-xl" />,
      path: `/user/dashboard/${userId}/workflow`,
      description: 'Create and manage data cleaning workflows'
    },
    {
      id: 'export',
      name: 'Export',
      icon: <FaDownload className="text-xl" />,
      path: `/user/dashboard/${userId}/export`,
      description: 'Export cleaned datasets'
    },
    {
      id: 'settings',
      name: 'Settings',
      icon: <FaCog className="text-xl" />,
      path: `/user/dashboard/${userId}/settings`,
      description: 'Configure cleaning parameters'
    }
  ];
};

export const GalaxyLayout: React.FC<GalaxyLayoutProps> = ({ children }) => {
  const location = useLocation();
  const { userId } = useParams();
  const tools = createTools(userId || '');
  const navigate = useNavigate();
  const { isAuthenticated, logout, user } = useAuth();
  // Détecte le tool actif à partir de l'URL
  const getActiveToolFromLocation = () => {
    const pathParts = location.pathname.split('/');
    // Cherche un id de tool principal dans l'URL
    for (const tool of tools) {
      if (pathParts.includes(tool.id)) return tool.id;
    }
    return null;
  };
  const [activeTool, setActiveTool] = useState<string | null>(getActiveToolFromLocation());
  const [showHistorySidebar, setShowHistorySidebar] = useState(true);
  // Output sidebar state removed
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const initials = (
    ((user?.first_name?.[0] || '') + (user?.last_name?.[0] || '')).toUpperCase() ||
    (user?.email?.[0]?.toUpperCase() || 'U')
  );

  // Synchronise activeTool avec l'URL
  useEffect(() => {
    setActiveTool(getActiveToolFromLocation());
  }, [location.pathname]);
  
  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, []);
  
  // Output sidebar integration removed
  
  // Fonction pour basculer la visibilité de la barre d'historique
  const toggleHistorySidebar = () => {
    setShowHistorySidebar(!showHistorySidebar);
  };
  
  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900 relative">
      <div className="w-64 bg-white dark:bg-gray-900 border-r dark:border-gray-700 flex flex-col sticky top-0 h-screen">
        <div className="px-4 py-3 border-b dark:border-gray-700 flex items-center gap-2">
          <img src={Logo} alt="Data Cleaning" className="h-6 w-6" />
          <span className="font-semibold text-gray-900 dark:text-white">Data Cleaning</span>
        </div>
        <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 px-4 py-2">Menu</div>
        <nav className="flex-1 overflow-y-auto p-2">
          {tools.map((tool) => (
            <React.Fragment key={tool.id}>
              <div className="pl-4">
                <Link
                  to={{ pathname: tool.path, search: location.search }}
                  className={`flex items-center gap-3 px-4 py-2 rounded-md text-sm transition-colors ${
                    (location.pathname.includes(tool.id) || activeTool === tool.id)
                      ? 'bg-[#ffcdd2] text-red-700 dark:bg-red-900/40 dark:text-red-200'
                      : 'text-gray-700 dark:text-gray-300 hover:bg-[#ffcdd2]/60 hover:text-red-700 dark:hover:bg-red-900/30 dark:hover:text-red-200'
                  }`}
                  onClick={(e) => {
                    setActiveTool(tool.id === activeTool ? null : tool.id);
                    // Prevent the link from navigating if we're just toggling the menu
                    if (tool.subTools && tool.subTools.length > 0) {
                      e.preventDefault();
                    }
                  }}
                >
                  <span className={`text-xl ${(location.pathname.includes(tool.id) || activeTool === tool.id) ? 'text-red-600 dark:text-red-300' : 'text-gray-500 dark:text-gray-400'}`}>
                    {tool.icon}
                  </span>
                  <div className="flex-1 text-sm font-medium">{tool.name}</div>
                </Link>
                {tool.subTools && activeTool === tool.id && (
                  <div className="pl-6">
                    {tool.subTools.map((subTool) => (
                      <Link
                        key={subTool.id}
                        to={{ pathname: subTool.path, search: location.search }}
                        className={`flex items-center gap-3 px-4 py-2 rounded-md text-sm transition-colors ${
                          location.pathname.includes(subTool.id)
                            ? 'bg-[#fffde7] text-red-700 dark:bg-yellow-900/30 dark:text-yellow-200'
                            : 'text-gray-700 dark:text-gray-300 hover:bg-[#fffde7] hover:text-red-700 dark:hover:bg-yellow-900/20 dark:hover:text-yellow-200'
                        }`}
                      >
                        <span className={`text-lg ${location.pathname.includes(subTool.id) ? 'text-red-600 dark:text-yellow-200' : 'text-gray-500 dark:text-gray-400'}`}>
                          {subTool.icon}
                        </span>
                        <div className="flex-1 text-sm font-medium">{subTool.name}</div>
                      </Link>
                    ))}
                  </div>
                )}
              </div>
              <div className="border-b border-gray-200 dark:border-gray-700 my-2" />
            </React.Fragment>
          ))}
        </nav>
      </div>
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="h-14 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-4 flex items-center justify-end">
          <div className="flex items-center space-x-4">
            <ThemeToggle />
            {isAuthenticated && (
              <div className="relative" ref={menuRef}>
                <button
                  onClick={() => setMenuOpen((v) => !v)}
                  className="inline-flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium text-gray-900 dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  <span className="flex h-8 w-8 items-center justify-center rounded-full bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 text-xs font-semibold">
                    {initials}
                  </span>
                  <span className="hidden sm:inline">{user?.first_name || 'Profile'}</span>
                  <svg className="h-4 w-4 opacity-70" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.084l3.71-3.853a.75.75 0 111.08 1.04l-4.24 4.4a.75.75 0 01-1.08 0l-4.24-4.4a.75.75 0 01.02-1.06z" clipRule="evenodd" />
                  </svg>
                </button>
                {menuOpen && (
                  <div className="absolute right-0 mt-2 w-44 rounded-md border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-lg py-1 z-50">
                    <Link
                      to="/profile"
                      onClick={() => setMenuOpen(false)}
                      className="block w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700"
                    >
                      Profile
                    </Link>
                    <button
                      onClick={() => { setMenuOpen(false); handleLogout(); }}
                      className="block w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700"
                    >
                      Logout
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
        <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 p-6 overflow-y-auto">
          <div className="h-full w-full">
            {children}
          </div>
        </div>
        
        <div className="relative">
          <button
            onClick={toggleHistorySidebar}
            className={`absolute top-1/2 -left-4 transform -translate-y-1/2 bg-white dark:bg-gray-800 p-2 rounded-l-lg shadow-md z-10 ${showHistorySidebar ? 'hidden' : 'block'}`}
            aria-label={showHistorySidebar ? 'Masquer l\'historique' : 'Afficher l\'historique'}
          >
            <FaHistory className="text-blue-500 text-xl" />
          </button>
          
          <div className={`h-full transition-all duration-300 ${showHistorySidebar ? 'w-80' : 'w-0'}`}>
            {showHistorySidebar && (
              <div className="h-full overflow-hidden">
                <DatasetHistorySidebar 
                  isOpen={showHistorySidebar} 
                  onToggle={toggleHistorySidebar} 
                />
              </div>
            )}
          </div>
        </div>
        </div>
      </div>
      
      {/* Output sidebar removed */}
    </div>
  );
};
