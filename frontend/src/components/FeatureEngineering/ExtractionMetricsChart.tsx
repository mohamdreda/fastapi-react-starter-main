import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';

interface ExtractionMetricsChartProps {
  algorithm: 'pca' | 'isomap';
  metrics?: {
    explained_variance_ratio?: number[];
    cumulative_explained_variance?: number[];
    reconstruction_error?: number;
    stress?: number;
    kl_divergence?: number;
    [key: string]: any;
  };
}

const ExtractionMetricsChart: React.FC<ExtractionMetricsChartProps> = ({ algorithm, metrics }) => {
  if (!metrics) return null;

  // For PCA, we show explained variance ratio by component
  if (algorithm === 'pca' && metrics.explained_variance_ratio) {
    const data = metrics.explained_variance_ratio.map((value, index) => ({
      component: `Component ${index + 1}`,
      explained_variance: value,
      cumulative_variance: metrics.cumulative_explained_variance?.[index] || 0,
    }));

    return (
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <h3 className="text-lg font-medium mb-4">Explained Variance by Component</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="component" />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(value: any) => [Number(value).toFixed(4), '']} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="explained_variance" 
              stroke="#8884d8" 
              name="Explained Variance Ratio" 
              strokeWidth={2}
              dot={{ r: 4 }}
            />
            <Line 
              type="monotone" 
              dataKey="cumulative_variance" 
              stroke="#82ca9d" 
              name="Cumulative Explained Variance" 
              strokeWidth={2}
              dot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }

  // For ISOMAP, we show stress or other metrics if available
  if (algorithm === 'isomap') {
    // If we have stress values by iteration (ideal case)
    if (metrics.stress_by_iteration) {
      const data = metrics.stress_by_iteration.map((value: number, index: number) => ({
        iteration: index + 1,
        stress: value,
      }));

      return (
        <div className="bg-white p-4 rounded-lg shadow mb-6">
          <h3 className="text-lg font-medium mb-4">ISOMAP Stress by Iteration</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="iteration" />
              <YAxis />
              <Tooltip formatter={(value: any) => [Number(value).toFixed(6), '']} />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="stress" 
                stroke="#8884d8" 
                name="Stress Value" 
                strokeWidth={2}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      );
    }

    // If we only have a single stress value, show it as a metric card
    if (metrics.stress || metrics.kl_divergence) {
      return (
        <div className="bg-white p-4 rounded-lg shadow mb-6">
          <h3 className="text-lg font-medium mb-4">ISOMAP Quality Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {metrics.stress && (
              <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                <div className="text-sm text-blue-600 mb-1">Stress Value</div>
                <div className="text-2xl font-semibold">{Number(metrics.stress).toFixed(6)}</div>
                <div className="text-xs text-gray-500 mt-1">Lower values indicate better embedding quality</div>
              </div>
            )}
            {metrics.kl_divergence && (
              <div className="bg-purple-50 p-4 rounded-lg border border-purple-100">
                <div className="text-sm text-purple-600 mb-1">KL Divergence</div>
                <div className="text-2xl font-semibold">{Number(metrics.kl_divergence).toFixed(6)}</div>
                <div className="text-xs text-gray-500 mt-1">Measures information loss in the embedding</div>
              </div>
            )}
          </div>
        </div>
      );
    }
  }

  // Fallback for when we don't have specific metrics to show
  return (
    <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 mb-6">
      <h3 className="text-lg font-medium mb-2">Algorithm Performance</h3>
      <p className="text-gray-600">
        {algorithm === 'pca' 
          ? 'PCA reduces dimensionality by finding orthogonal components that maximize variance.'
          : 'ISOMAP preserves the geodesic distances between points in the lower-dimensional space.'}
      </p>
      {metrics.reconstruction_error && (
        <div className="mt-3 p-3 bg-blue-50 rounded-lg">
          <span className="font-medium">Reconstruction Error:</span> {Number(metrics.reconstruction_error).toFixed(6)}
        </div>
      )}
    </div>
  );
};

export default ExtractionMetricsChart;
