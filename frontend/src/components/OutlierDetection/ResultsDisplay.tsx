import React, { useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface VisualizationData {
  reduced_features?: boolean;
  scatter_plot_path?: string;
}

interface OutlierResults {
  total_samples?: number;
  outlier_count?: number;
  processing_time?: number;
  metrics?: Record<string, number>;
  outlier_indices?: number[];
  outlier_scores?: number[];
  cluster_labels?: number[];
  visualization_data?: VisualizationData;
}

interface ResultsDisplayProps {
  results: OutlierResults;
  config: any; 
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results, config }) => {
  const [activeTab, setActiveTab] = useState('summary');

  const isPipelineConfig = config && config.featureExtraction;

  const pipelineDescription = isPipelineConfig ? {
    featureExtraction: {
      algorithm: config.featureExtraction.algorithm,
      params: Object.entries(config.featureExtraction.parameters)
        .filter(([key]) => key.startsWith(config.featureExtraction.algorithm) || ['latent_dim', 'pcaComponents', 'epochs', 'batchSize'].includes(key))
        .map(([key, value]) => `${key.replace(/_/g, ' ')}: ${value}`)
    },
    clustering: {
      algorithm: config.clustering.algorithm,
      params: Object.entries(config.clustering.parameters)
        .filter(([key]) => key.startsWith(config.clustering.algorithm) || ['eps', 'minSamples', 'denclueH', 'denclueEps'].includes(key))
        .map(([key, value]) => `${key.replace(/_/g, ' ')}: ${value}`)
    },
    anomalyDetection: {
      algorithm: config.anomalyDetection.algorithm,
      params: Object.entries(config.anomalyDetection.parameters)
        .filter(([key]) => key.startsWith(config.anomalyDetection.algorithm.split('_')[0]) || ['contamination', 'nEstimators', 'maxSamples', 'lofNeighbors', 'lofContamination', 'ocsvm_nu', 'ocsvm_kernel', 'ocsvm_gamma'].includes(key))
        .map(([key, value]) => `${key.replace(/_/g, ' ')}: ${value}`)
    }
  } : {
    featureExtraction: {
      algorithm: 'autoencoder',
      params: Object.entries(config || {}).map(([key, value]) => `${key.replace(/_/g, ' ')}: ${value}`)
    },
    clustering: { algorithm: 'N/A', params: [] },
    anomalyDetection: { algorithm: 'N/A', params: [] }
  };

  const formatAlgorithmName = (id: string) => {
    const names: Record<string, string> = {
      'autoencoder': 'Autoencoder',
      'pca': 'PCA',
      'isomap': 'ISOMAP',
      'dbscan': 'DBSCAN',
      'denclue': 'DENCLUE',
      'optics': 'OPTICS',
      'isolation_forest': 'Isolation Forest',
      'lof': 'Local Outlier Factor',
      'one_class_svm': 'One-Class SVM',
      'ocsvm': 'One-Class SVM'
    };
    return names[id] || id;
  };

  const renderSummaryTab = () => {
    if (!results) return <div>No results available</div>;
    
    return (
      <div className="space-y-6">
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-medium mb-3">Pipeline Configuration</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm font-medium">Feature Extraction</p>
              <p className="mb-1">{formatAlgorithmName(pipelineDescription.featureExtraction.algorithm)}</p>
              <ul className="text-sm text-gray-600">
                {pipelineDescription.featureExtraction.params.map((param, i) => (
                  <li key={i}>{param}</li>
                ))}
              </ul>
            </div>
            {isPipelineConfig && (
              <>
                <div>
                  <p className="text-sm font-medium">Clustering</p>
                  <p className="mb-1">{formatAlgorithmName(pipelineDescription.clustering.algorithm)}</p>
                  <ul className="text-sm text-gray-600">
                    {pipelineDescription.clustering.params.map((param, i) => (
                      <li key={i}>{param}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="text-sm font-medium">Anomaly Detection</p>
                  <p className="mb-1">{formatAlgorithmName(pipelineDescription.anomalyDetection.algorithm)}</p>
                  <ul className="text-sm text-gray-600">
                    {pipelineDescription.anomalyDetection.params.map((param, i) => (
                      <li key={i}>{param}</li>
                    ))}
                  </ul>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="font-medium mb-3">Run Metrics</h3>
            <ul className="text-sm text-gray-700 space-y-2">
              <li>Total Samples: <span className="font-semibold">{results.total_samples ?? 'N/A'}</span></li>
              <li>Outliers Detected: <span className="font-semibold text-red-600">{results.outlier_count ?? 'N/A'}</span></li>
              <li>Processing Time: <span className="font-semibold">{results.processing_time?.toFixed(2) ?? 'N/A'}s</span></li>
            </ul>
          </div>
          
          {results.metrics && Object.keys(results.metrics).length > 0 && (
            <div className="bg-white p-4 rounded-lg shadow">
              <h3 className="font-medium mb-3">Evaluation Metrics</h3>
              {results.metrics.source && typeof results.metrics.source === 'string' && (
                <p className="text-xs text-gray-500 mb-2">
                  Source: {results.metrics.source === 'original_dataset' ? 'Original dataset Class column' : 'Provided ground truth'}
                </p>
              )}
              <ul className="text-sm text-gray-700 space-y-2">
                {Object.entries(results.metrics).map(([key, value]) => {
                  // Skip non-numeric or special fields
                  if (typeof value !== 'number' || key === 'source') return null;
                  
                  // Format the metric name for display
                  const formattedKey = key.replace(/_/g, ' ');
                  
                  // Format the value based on the metric type
                  let formattedValue;
                  if (key === 'auc_roc' || key === 'average_precision' || 
                      key === 'precision' || key === 'recall' || key === 'f1_score' || 
                      key === 'accuracy') {
                    formattedValue = value.toFixed(4);
                  } else {
                    formattedValue = typeof value === 'number' ? value.toFixed(2) : value;
                  }
                  
                  return (
                    <li key={key}>
                      <span className="capitalize">{formattedKey}</span>: 
                      <span className="font-semibold">{formattedValue}</span>
                    </li>
                  );
                })}
              </ul>
              {config?.anomalyDetection?.algorithm === 'one_class_svm' && (
                <div className="mt-3 text-xs text-gray-600">
                  <p>One-Class SVM metrics are calculated using the original dataset's Class column as ground truth.</p>
                </div>
              )}
            </div>
          )}
        </div>

        {(results.visualization_data?.scatter_plot_path || (results as any)?.scatter_plot_path) && (
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="font-medium mb-3">Outlier Visualization</h3>
            <img 
              src={((): string => {
                const p = results.visualization_data?.scatter_plot_path || (results as any)?.scatter_plot_path;
                if (!p) return '';
                if (p.startsWith('data:image/')) return p;
                if (p.length > 100 && /^[A-Za-z0-9+/=]+$/.test(p)) return `data:image/png;base64,${p}`;
                return p.startsWith('/') ? `${API_BASE_URL}${p}` : p;
              })()} 
              alt="Outlier Visualization" 
              className="w-full h-auto rounded-lg border" 
            />
          </div>
        )}
      </div>
    );
  };

  const renderDetailedTab = () => {
    if (!results || !results.outlier_indices || !results.outlier_scores) {
      return <div>No detailed results available</div>;
    }

    const detailedData = results.outlier_indices.map((original_index, i) => ({
      original_index,
      outlier_score: results.outlier_scores ? results.outlier_scores[i] : 'N/A',
      cluster_label: results.cluster_labels ? results.cluster_labels[i] : 'N/A',
    }));

    return (
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="font-medium mb-3">Detailed Outlier Information</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Original Index</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Outlier Score</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Cluster Label</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {detailedData.map((item, index) => (
                <tr key={index}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{item.original_index}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{typeof item.outlier_score === 'number' ? item.outlier_score.toFixed(4) : item.outlier_score}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.cluster_label}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderExportTab = () => {
    const handleExportJSON = () => {
      const exportData = {
        results,
        config
      };
      const dataStr = JSON.stringify(exportData, null, 2);
      const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`;
      const exportFileName = `outlier_detection_results_${new Date().toISOString().slice(0, 10)}.json`;
      
      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', exportFileName);
      linkElement.click();
    };

    const handleExportCSV = () => {
      if (!results || !results.outlier_indices) return;
      
      let csvContent = "data:text/csv;charset=utf-8,";
      csvContent += "Original Index,Outlier Score,Cluster Label\r\n";
      
      results.outlier_indices.forEach((original_index, i) => {
        const score = results.outlier_scores ? results.outlier_scores[i] : 'N/A';
        const label = results.cluster_labels ? results.cluster_labels[i] : 'N/A';
        csvContent += `${original_index},${score},${label}\r\n`;
      });
      
      const encodedUri = encodeURI(csvContent);
      const exportFileDefaultName = `outlier_detection_results_${new Date().toISOString().slice(0, 10)}.csv`;
      
      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', encodedUri);
      linkElement.setAttribute('download', exportFileDefaultName);
      linkElement.click();
    };
    
    return (
      <div className="space-y-6">
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="font-medium mb-3">Export Results</h3>
          <p className="text-gray-600 mb-4">
            Export the outlier detection results in different formats for further analysis or reporting.
          </p>
          <div className="flex space-x-4">
            <button
              onClick={handleExportJSON}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Export as JSON
            </button>
            <button
              onClick={handleExportCSV}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              Export as CSV
            </button>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="font-medium mb-3">Pipeline Configuration Export</h3>
          <p className="text-gray-600 mb-4">
            Export the current pipeline configuration to reuse in future runs.
          </p>
          <button
            onClick={() => {
              const dataStr = JSON.stringify(config, null, 2);
              const dataUri = `data:application/json;charset=utf-8,${encodeURIComponent(dataStr)}`;
              const exportFileName = `pipeline_config_${new Date().toISOString().slice(0, 10)}.json`;
              
              const linkElement = document.createElement('a');
              linkElement.setAttribute('href', dataUri);
              linkElement.setAttribute('download', exportFileName);
              linkElement.click();
            }}
            className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
          >
            Export Configuration
          </button>
        </div>
      </div>
    );
  };
  
  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-6">Outlier Detection Results</h2>
      
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex -mb-px">
          <button
            className={`py-2 px-4 border-b-2 font-medium text-sm ${activeTab === 'summary' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}
            onClick={() => setActiveTab('summary')}
          >
            Summary
          </button>
          <button
            className={`ml-8 py-2 px-4 border-b-2 font-medium text-sm ${activeTab === 'detailed' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}
            onClick={() => setActiveTab('detailed')}
          >
            Detailed Results
          </button>
          <button
            className={`ml-8 py-2 px-4 border-b-2 font-medium text-sm ${activeTab === 'export' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}
            onClick={() => setActiveTab('export')}
          >
            Export
          </button>
        </nav>
      </div>
      
      <div>
        {activeTab === 'summary' && renderSummaryTab()}
        {activeTab === 'detailed' && renderDetailedTab()}
        {activeTab === 'export' && renderExportTab()}
      </div>
    </div>
  );
};

export default ResultsDisplay;
