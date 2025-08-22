import React, { useState } from 'react';

interface AlgorithmPipelineProps {
  config: any;
  onAlgorithmChange: (step: string, algorithm: string) => void;
  onParameterChange: (step: string, paramName: string, value: any) => void;
  onGeneralParameterChange: (paramName: string, value: any) => void;
  onRun: () => void;
  loading: boolean;
}

export const AlgorithmPipeline: React.FC<AlgorithmPipelineProps> = ({ 
  config, 
  onAlgorithmChange, 
  onParameterChange, 
  onGeneralParameterChange,
  onRun,
  loading
}) => {
  const [activeStep, setActiveStep] = useState('featureExtraction');
  
  // Algorithm options for each step
  const algorithms = {
    featureExtraction: [
      { id: 'autoencoder', name: 'Autoencoder' },
      { id: 'pca', name: 'PCA (Principal Component Analysis)' },
      { id: 'isomap', name: 'ISOMAP' }
    ],
    clustering: [
      { id: 'dbscan', name: 'DBSCAN' },
      { id: 'denclue', name: 'DENCLUE' },
      { id: 'optics', name: 'OPTICS' }
    ],
    anomalyDetection: [
      { id: 'isolation_forest', name: 'Isolation Forest' },
      { id: 'lof', name: 'Local Outlier Factor (LOF)' },
      { id: 'one_class_svm', name: 'One-Class SVM' }
    ]
  };
  
  // Render number input field
  const renderNumberInput = (label: string, paramName: string, min: number, max: number, step: number = 1) => (
    <div className="mb-3">
      <label className="block text-sm font-medium mb-1">{label}</label>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={config[activeStep].parameters[paramName]}
        onChange={(e) => onParameterChange(activeStep, paramName, parseFloat(e.target.value))}
        className="w-full p-2 border rounded"
      />
    </div>
  );
  
  // Render select input field
  const renderSelectInput = (label: string, paramName: string, options: { value: string, label: string }[]) => (
    <div className="mb-3">
      <label className="block text-sm font-medium mb-1">{label}</label>
      <select
        value={config[activeStep].parameters[paramName]}
        onChange={(e) => onParameterChange(activeStep, paramName, e.target.value)}
        className="w-full p-2 border rounded"
      >
        {options.map(option => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
  
  // Render parameters for the selected algorithm
  const renderParameters = () => {
    switch (activeStep) {
      case 'featureExtraction':
        switch (config.featureExtraction.algorithm) {
          case 'autoencoder':
            return (
              <>
                {renderNumberInput('Latent Dimension', 'latent_dim', 2, 100)}
                {renderNumberInput('Epochs', 'autoencoder_epochs', 10, 200)}
                {renderNumberInput('Batch Size', 'autoencoder_batch_size', 8, 256)}
              </>
            );
          case 'pca':
            return (
              <>
                {renderNumberInput('Number of Components', 'pca_n_components', 2, 100)}
              </>
            );
          case 'isomap':
            return (
              <>
                {renderNumberInput('Number of Components', 'isomap_n_components', 2, 100)}
                {renderNumberInput('Number of Neighbors', 'isomap_n_neighbors', 2, 50)}
              </>
            );
        }
        break;
        
      case 'clustering':
        switch (config.clustering.algorithm) {
          case 'dbscan':
            return (
              <>
                {renderNumberInput('Epsilon (eps)', 'clustering_eps', 0.01, 2, 0.01)}
                {renderNumberInput('Min Samples', 'clustering_min_samples', 2, 50)}
              </>
            );
          case 'denclue':
            return (
              <>
                {renderNumberInput('Bandwidth (h)', 'denclue_h', 0.01, 1, 0.01)}
                {renderNumberInput('Epsilon', 'denclue_eps', 0.0001, 0.1, 0.0001)}
              </>
            );
          case 'optics':
            return (
              <>
                {renderNumberInput('Min Samples', 'optics_min_samples', 2, 50)}
                {renderNumberInput('Max Epsilon', 'optics_max_eps', 0.1, 20, 0.1)}
                {renderNumberInput('Xi', 'optics_xi', 0.01, 0.5, 0.01)}
              </>
            );
        }
        break;
        
      case 'anomalyDetection':
        switch (config.anomalyDetection.algorithm) {
          case 'isolation_forest':
            return (
              <>
                {renderNumberInput('Number of Estimators', 'if_n_estimators', 50, 500)}
                {renderSelectInput('Contamination', 'if_contamination', [
                  { value: 'auto', label: 'Auto' },
                  { value: '0.01', label: '0.01 (1%)' },
                  { value: '0.05', label: '0.05 (5%)' },
                  { value: '0.1', label: '0.1 (10%)' },
                  { value: '0.2', label: '0.2 (20%)' },
                ])}
                {renderSelectInput('Max Samples', 'if_max_samples', [
                  { value: 'auto', label: 'Auto' },
                  { value: '100', label: '100' },
                  { value: '256', label: '256' },
                  { value: '512', label: '512' },
                  { value: '1024', label: '1024' },
                ])}
              </>
            );
          case 'lof':
            return (
              <>
                {renderNumberInput('Number of Neighbors', 'lof_n_neighbors', 5, 100)}
                {renderSelectInput('Contamination', 'lof_contamination', [
                  { value: 'auto', label: 'Auto' },
                  { value: '0.01', label: '0.01 (1%)' },
                  { value: '0.05', label: '0.05 (5%)' },
                  { value: '0.1', label: '0.1 (10%)' },
                  { value: '0.2', label: '0.2 (20%)' },
                ])}
              </>
            );
          case 'one_class_svm':
            return (
              <>
                {renderNumberInput('Nu', 'ocsvm_nu', 0.01, 0.5, 0.01)}
                {renderSelectInput('Kernel', 'ocsvm_kernel', [
                  { value: 'rbf', label: 'RBF' },
                  { value: 'linear', label: 'Linear' },
                  { value: 'poly', label: 'Polynomial' },
                  { value: 'sigmoid', label: 'Sigmoid' },
                ])}
                {renderSelectInput('Gamma', 'ocsvm_gamma', [
                  { value: 'scale', label: 'Scale' },
                  { value: 'auto', label: 'Auto' },
                ])}
              </>
            );
        }
        break;
        
      case 'general':
        return (
          <>
            <div className="mb-3">
              <label className="block text-sm font-medium mb-1">Random State</label>
              <input
                type="number"
                min={0}
                max={1000}
                value={config.general.random_state}
                onChange={(e) => onGeneralParameterChange('random_state', parseInt(e.target.value))}
                className="w-full p-2 border rounded"
              />
            </div>
            <div className="mb-3">
              <label className="block text-sm font-medium mb-1">Evaluation Type</label>
              <select
                value={config.general.evaluation_type}
                onChange={(e) => onGeneralParameterChange('evaluation_type', e.target.value)}
                className="w-full p-2 border rounded"
              >
                <option value="auto">Auto-detect</option>
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>
            </div>
          </>
        );
    }
  };
  
  return (
    <div className="bg-white p-6 rounded-lg shadow mb-8">
      <h2 className="text-xl font-semibold mb-6">Configure Outlier Detection Pipeline</h2>
      
      {/* Pipeline Steps Visualization */}
      <div className="flex items-center mb-8">
        <div 
          className={`flex-1 p-3 rounded-l-lg cursor-pointer text-center ${
            activeStep === 'featureExtraction' ? 'bg-blue-600 text-white' : 'bg-gray-200'
          }`}
          onClick={() => setActiveStep('featureExtraction')}
        >
          <div className="font-medium">Step 1</div>
          <div>Feature Extraction</div>
        </div>
        <div className="w-8 h-1 bg-gray-300"></div>
        <div 
          className={`flex-1 p-3 cursor-pointer text-center ${
            activeStep === 'clustering' ? 'bg-blue-600 text-white' : 'bg-gray-200'
          }`}
          onClick={() => setActiveStep('clustering')}
        >
          <div className="font-medium">Step 2</div>
          <div>Clustering</div>
        </div>
        <div className="w-8 h-1 bg-gray-300"></div>
        <div 
          className={`flex-1 p-3 cursor-pointer text-center ${
            activeStep === 'anomalyDetection' ? 'bg-blue-600 text-white' : 'bg-gray-200'
          }`}
          onClick={() => setActiveStep('anomalyDetection')}
        >
          <div className="font-medium">Step 3</div>
          <div>Anomaly Detection</div>
        </div>
        <div className="w-8 h-1 bg-gray-300"></div>
        <div 
          className={`flex-1 p-3 rounded-r-lg cursor-pointer text-center ${
            activeStep === 'general' ? 'bg-blue-600 text-white' : 'bg-gray-200'
          }`}
          onClick={() => setActiveStep('general')}
        >
          <div className="font-medium">Step 4</div>
          <div>General Settings</div>
        </div>
      </div>
      
      {/* Current Pipeline Configuration Summary */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="font-medium mb-3">Current Pipeline Configuration</h3>
        <div className="grid grid-cols-3 gap-4">
          <div>
            <p className="text-sm font-medium">Feature Extraction</p>
            <p>{algorithms.featureExtraction.find(a => a.id === config.featureExtraction.algorithm)?.name}</p>
          </div>
          <div>
            <p className="text-sm font-medium">Clustering</p>
            <p>{algorithms.clustering.find(a => a.id === config.clustering.algorithm)?.name}</p>
          </div>
          <div>
            <p className="text-sm font-medium">Anomaly Detection</p>
            <p>{algorithms.anomalyDetection.find(a => a.id === config.anomalyDetection.algorithm)?.name}</p>
          </div>
        </div>
      </div>
      
      {/* Algorithm Selection and Parameters */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Algorithm Selection */}
        <div>
          <h3 className="font-medium mb-4">
            {activeStep === 'featureExtraction' ? 'Select Feature Extraction Algorithm' :
             activeStep === 'clustering' ? 'Select Clustering Algorithm' :
             activeStep === 'anomalyDetection' ? 'Select Anomaly Detection Algorithm' :
             'General Settings'}
          </h3>
          
          {activeStep !== 'general' && (
            <div className="space-y-3">
              {algorithms[activeStep as keyof typeof algorithms].map(algorithm => (
                <label 
                  key={algorithm.id} 
                  className="flex items-center p-3 border rounded hover:bg-gray-50 cursor-pointer"
                >
                  <input
                    type="radio"
                    name={`${activeStep}Algorithm`}
                    value={algorithm.id}
                    checked={config[activeStep].algorithm === algorithm.id}
                    onChange={() => onAlgorithmChange(activeStep, algorithm.id)}
                    className="mr-3"
                  />
                  <div>
                    <div className="font-medium">{algorithm.name}</div>
                    <div className="text-sm text-gray-600">
                      {algorithm.id === 'autoencoder' && 'Neural network for dimensionality reduction'}
                      {algorithm.id === 'pca' && 'Linear dimensionality reduction'}
                      {algorithm.id === 'isomap' && 'Non-linear dimensionality reduction'}
                      {algorithm.id === 'dbscan' && 'Density-based spatial clustering'}
                      {algorithm.id === 'denclue' && 'Density-based clustering using kernel density estimation'}
                      {algorithm.id === 'optics' && 'Ordering points to identify clustering structure'}
                      {algorithm.id === 'isolation_forest' && 'Isolates outliers by random partitioning'}
                      {algorithm.id === 'lof' && 'Identifies outliers by measuring local deviation'}
                      {algorithm.id === 'one_class_svm' && 'One-class classification using support vector machines'}
                    </div>
                  </div>
                </label>
              ))}
            </div>
          )}
        </div>
        
        {/* Parameters Configuration */}
        <div>
          <h3 className="font-medium mb-4">
            {activeStep === 'featureExtraction' ? 'Configure Feature Extraction Parameters' :
             activeStep === 'clustering' ? 'Configure Clustering Parameters' :
             activeStep === 'anomalyDetection' ? 'Configure Anomaly Detection Parameters' :
             'Configure General Settings'}
          </h3>
          
          <div className="p-4 border rounded">
            {renderParameters()}
          </div>
        </div>
      </div>
      
      {/* Run Button */}
      <div className="mt-8 flex justify-end">
        <button
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
          onClick={onRun}
          disabled={loading}
        >
          {loading ? 'Processing...' : 'Run Outlier Detection'}
        </button>
      </div>
    </div>
  );
};

export default AlgorithmPipeline;
