import React from 'react';
import { Link, useParams, useLocation } from 'react-router-dom';

export const FeatureEngineeringPage: React.FC = () => {
  const { userId } = useParams<{ userId: string }>();
  const { search } = useLocation();

  const featurePages = [
    {
      name: 'Feature Extraction Pipeline',
      description: 'Build a custom pipeline with multiple algorithms for feature extraction, clustering, and anomaly detection.',
      path: `/user/dashboard/${userId}/feature-engineering/extraction`
    },
    {
      name: 'Feature Selection',
      description: 'Select the most relevant features for your model.',
      path: `/user/dashboard/${userId}/feature-engineering/selection`
    },
    {
      name: 'Autoencoder Anomaly Detection',
      description: 'Use a dedicated autoencoder model for specialized feature engineering and anomaly detection.',
      path: `/user/dashboard/${userId}/feature-engineering/autoencoder`
    }
  ];

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6">Feature Engineering</h1>
      <p className="mb-8 text-lg text-gray-600">
        Select a feature engineering method to begin. Each method offers a different approach to transforming your data and discovering insights.
      </p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {featurePages.map((page, index) => (
          <Link to={`${page.path}${search || ''}`} key={index} className="block p-6 bg-white border rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300">
            <h2 className="text-2xl font-semibold mb-3 text-blue-600">{page.name}</h2>
            <p className="text-gray-700">{page.description}</p>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default FeatureEngineeringPage;
