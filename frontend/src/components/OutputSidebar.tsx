import React, { useState } from 'react';
import { FaChevronLeft, FaChevronRight, FaTimes } from 'react-icons/fa';
import { motion } from 'framer-motion';

export interface Output {
  type: 'upload' | 'cleaning' | 'feature' | 'transformation' | 'model' | 'report';
  title: string;
  content: string;
  timestamp: string;
  details?: any;
}

interface OutputSidebarProps {
  outputs: Output[];
  currentStep: string;
  onClose?: () => void;
}

const OutputSidebar: React.FC<OutputSidebarProps> = ({ outputs, currentStep, onClose }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const formatOutputContent = (output: Output) => {
    switch (output.type) {
      case 'upload':
        return (
          <div>
            <p className="text-sm">Dataset: {output.details?.name}</p>
            <p className="text-sm">Size: {output.details?.size} MB</p>
            <p className="text-sm">Rows: {output.details?.rows}</p>
            <p className="text-sm">Columns: {output.details?.columns}</p>
          </div>
        );
      case 'cleaning':
        return (
          <div>
            <p className="text-sm">Missing values: {output.details?.missingValues}</p>
            <p className="text-sm">Duplicates: {output.details?.duplicates}</p>
            <p className="text-sm">Outliers: {output.details?.outliers}</p>
          </div>
        );
      case 'feature':
        return (
          <div>
            <p className="text-sm">New features: {output.details?.newFeatures}</p>
            <p className="text-sm">Feature type: {output.details?.featureType}</p>
            <p className="text-sm">Transformation: {output.details?.transformation}</p>
          </div>
        );
      case 'transformation':
        return (
          <div>
            <p className="text-sm">Normalization: {output.details?.normalization}</p>
            <p className="text-sm">Encoding: {output.details?.encoding}</p>
            <p className="text-sm">Scaling: {output.details?.scaling}</p>
          </div>
        );
      case 'model':
        return (
          <div>
            <p className="text-sm">Accuracy: {output.details?.accuracy}%</p>
            <p className="text-sm">Recall: {output.details?.recall}%</p>
            <p className="text-sm">Precision: {output.details?.precision}%</p>
          </div>
        );
      case 'report':
        return (
          <div>
            <p className="text-sm">Final metrics: {output.details?.metrics}</p>
            <p className="text-sm">Processing time: {output.details?.time}s</p>
          </div>
        );
      default:
        return <p className="text-sm">{output.content}</p>;
    }
  };

  return (
    <motion.div
      initial={{ width: isCollapsed ? '48px' : '320px' }}
      animate={{ width: isCollapsed ? '48px' : '320px' }}
      transition={{ duration: 0.3 }}
      className="fixed right-0 top-0 h-screen bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 border-l border-gray-200 dark:border-gray-700 shadow-lg overflow-y-auto z-40"
    >
      <div className={`flex items-center ${isCollapsed ? 'justify-center p-2' : 'justify-between p-4'} border-b dark:border-gray-700`}>
        {!isCollapsed && (
          <div className="flex flex-col">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Workflow Outputs</h3>
            {currentStep && (
              <span className="text-xs text-gray-500 dark:text-gray-400">Step: {currentStep}</span>
            )}
          </div>
        )}
        <div className={`flex ${isCollapsed ? '' : 'space-x-2'}`}>
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
            aria-label={isCollapsed ? 'Expand outputs' : 'Collapse outputs'}
          >
            {isCollapsed ? <FaChevronLeft /> : <FaChevronRight />}
          </button>
          {!isCollapsed && onClose && (
            <button
              onClick={onClose}
              className="p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
              aria-label="Close outputs"
            >
              <FaTimes />
            </button>
          )}
        </div>
      </div>

      {!isCollapsed && (
        <div className="p-4">
          {outputs.map((output, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-4 p-4 bg-white dark:bg-gray-900 rounded-lg shadow border border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium">{output.title}</h4>
                <span className="text-xs text-gray-500 dark:text-gray-400">{output.timestamp}</span>
              </div>
              {formatOutputContent(output)}
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  );
};

export default OutputSidebar;
