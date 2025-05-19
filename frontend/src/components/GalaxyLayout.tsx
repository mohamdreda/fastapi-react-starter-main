import React from 'react';
import { Link, useLocation, useParams } from 'react-router-dom';
import {
  FaTable,
  FaSearch,
  FaExclamationTriangle,
  FaRandom,
  FaChartBar,
  FaDownload,
  FaCog,
} from 'react-icons/fa';

interface Tool {
  id: string;
  name: string;
  icon: React.ReactNode;
  path: string;
  description: string;
}

const createTools = (userId: string): Tool[] => [
  {
    id: 'datasets',
    name: 'Datasets',
    icon: <FaTable />,
    path: `/user/dashboard/${userId}/datasets`,
    description: 'Manage your uploaded datasets'
  },

  {
    id: 'diagnosis',
    name: 'Data Diagnosis',
    icon: <FaSearch />,
    path: `/user/dashboard/${userId}/diagnosis`,
    description: 'Analyze data quality and statistics'
  },
  {
    id: 'outliers',
    name: 'Outliers',
    icon: <FaChartBar />,
    path: `/user/dashboard/${userId}/outliers`,
    description: 'Identify and handle outliers'
  },
  {
    id: 'imputation',
    name: 'Missing Values',
    icon: <FaExclamationTriangle />,
    path: `/user/dashboard/${userId}/imputation`,
    description: 'Handle missing values in your data'
  },
  {
    id: 'duplicates',
    name: 'Duplicates',
    icon: <FaRandom />,
    path: `/user/dashboard/${userId}/duplicates`,
    description: 'Detect and remove duplicate records'
  },
  {
    id: 'export',
    name: 'Export',
    icon: <FaDownload />,
    path: `/user/dashboard/${userId}/export`,
    description: 'Export cleaned datasets'
  },
  {
    id: 'settings',
    name: 'Settings',
    icon: <FaCog />,
    path: `/user/dashboard/${userId}/settings`,
    description: 'Configure cleaning parameters'
  }
];

interface GalaxyLayoutProps {
  children: React.ReactNode;
}

export const GalaxyLayout: React.FC<GalaxyLayoutProps> = ({ children }) => {
  const location = useLocation();
  const { userId } = useParams();
  const tools = createTools(userId || '');

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Left Sidebar - Tool Navigation */}
      <div className="w-64 bg-white shadow-lg">
          <div className="p-4 border-b">
            <h2 className="text-lg font-semibold"> </h2>
          </div>
        <nav className="p-2">
          {tools.map((tool, index) => {
            const isActive = location.pathname.includes(tool.id);
            return (
              <React.Fragment key={tool.id}>
                {index > 0 && (
                  <div className="border-b border-gray-200 my-2" />
                )}
                <Link
                  to={tool.path}
                  className={`flex items-center space-x-3 px-3 py-3 rounded-md transition-colors ${isActive
                    ? 'bg-primary-50 text-primary-600'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                    }`}
                >
                  <span className={`text-lg ${isActive ? 'text-primary-500' : 'text-gray-400'}`}>
                    {tool.icon}
                  </span>
                  <div className="flex-1">
                    <div className="text-sm font-medium">{tool.name}</div>
                    <div className="text-xs text-gray-500">{tool.description}</div>
                  </div>
                  {isActive && (
                    <div className="w-1.5 h-8 bg-primary-500 rounded-full ml-2"></div>
                  )}
                </Link>
              </React.Fragment>
            );
          })}
        </nav>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-auto">
        <div className="container mx-auto p-6">
          {children}
        </div>
      </div>
    </div>
  );
};
