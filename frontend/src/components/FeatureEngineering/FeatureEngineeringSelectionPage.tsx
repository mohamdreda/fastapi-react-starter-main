import React from 'react';
import { useParams, useSearchParams } from 'react-router-dom';

const FeatureEngineeringSelectionPage: React.FC = () => {
  const { userId } = useParams();
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get('session_id') || localStorage.getItem('active_session_id') || '';

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-2">Feature Selection</h1>
      <p className="text-gray-600 mb-6">
        Coming soon. This page will let you select the most relevant features from your dataset.
      </p>

      {sessionId && (
        <div className="mb-4 text-xs text-gray-500">
          Active session: <span className="font-mono">{sessionId}</span>
        </div>
      )}

      <div className="p-4 border rounded bg-white">
        <p className="text-sm text-gray-700">
          Placeholder component to keep navigation intact. Implementation will follow the established
          pattern used by <span className="font-semibold">Autoencoder</span> and <span className="font-semibold">Feature Extraction</span> pages
          (dataset selection, params, run, results, downloads, and session tracking).
        </p>
      </div>
    </div>
  );
};

export default FeatureEngineeringSelectionPage;
