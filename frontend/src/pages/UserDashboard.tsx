import { useEffect } from 'react';
import { useNavigate, useParams, Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { GalaxyLayout } from '@/components/GalaxyLayout';
import { WelcomeDashboard } from '@/components/WelcomeDashboard';
import { DatasetManager } from '@/pages/DatasetManager';
import { DiagnosisDashboard } from '@/components/DataDiagnosis/DiagnosisDashboard';
import FeatureEngineeringAutoencoderPage from '@/components/FeatureEngineering/FeatureEngineeringAutoencoderPage';
import FeatureEngineeringExtractionPage from '@/components/FeatureEngineering/FeatureEngineeringExtractionPage';
import ClusteringDensityPage from '../pages/ClusteringDensityPage';
import DataTransformationPage from '@/pages/DataTransformationPage';
import StandardizationPage from '@/pages/StandardizationPage';
import CategoricalTransformationPage from '@/pages/CategoricalTransformationPage';

// Import du composant d'imputation
import ImputationPage from './ImputationPage';

import DeduplicationPage from './DeduplicationPage';
import PreprocessingPage from './PreprocessingPage';
import BlockingPage from './BlockingPage';
import SimilarityCalculationPage from './SimilarityCalculationPage';
import ClassificationPage from './ClassificationPage';
import ClusteringPage from './ClusteringPage';
import ResultsResolutionPage from './ResultsResolutionPage';
import ProfilePage from '@/pages/ProfilePage';

import OutlierStatisticalPage from './OutlierStatisticalPage';
import OutlierMLPage from './OutlierMLPage';
import WorkflowPage from './Workflow';

import FeatureEngineeringSelectionPage from '@/components/FeatureEngineering/FeatureEngineeringSelectionPage';
import FeatureEngineeringPage from '@/pages/FeatureEngineeringPage';

const Export = () => (
  <div className="px-4 py-2">
    <h1 className="text-2xl font-bold mb-4">Export Cleaned Data</h1>
    <p>Select a dataset to export after cleaning.</p>
  </div>
);

// Settings route now renders the user Profile page (company, phone number)
const Settings = () => <ProfilePage />;

export default function UserDashboard() {
  const { userId } = useParams();
  const navigate = useNavigate();
  const { user } = useAuth();

  useEffect(() => {
    if (!user) {
      navigate('/login');
      return;
    }

    if (user.id.toString() !== userId) {
      navigate(`/user/dashboard/${user.id}`);
    }
  }, [user, userId, navigate]);

  if (!user) return null;

  return (
    <GalaxyLayout>
      <div className="min-h-screen flex flex-col">
        <Routes>
          <Route index element={<WelcomeDashboard />} />
          <Route path="datasets" element={<DatasetManager />} />
          <Route path="diagnosis">
            <Route index element={<DiagnosisDashboard />} />
            <Route path=":datasetId" element={<DiagnosisDashboard />} />
          </Route>
          <Route path="clustering/density" element={<ClusteringDensityPage />} />
          <Route path="outlier-detection" element={<Navigate to="statistical" replace />} />
          <Route path="outlier-detection/statistical" element={<OutlierStatisticalPage />} />
          <Route path="outlier-detection/ml" element={<OutlierMLPage />} />
          <Route path="imputation" element={<ImputationPage />} />
          <Route path="duplicates">
            <Route index element={<Navigate to={`data-dedup`} replace />} />
            <Route path="data-dedup" element={<DeduplicationPage />} />
            <Route path="preprocessing" element={<PreprocessingPage />} />
            <Route path="blocking" element={<BlockingPage />} />
            <Route path="similarity" element={<SimilarityCalculationPage />} />
            <Route path="classification" element={<ClassificationPage />} />
            <Route path="clustering" element={<ClusteringPage />} />
            <Route path="results" element={<ResultsResolutionPage />} />
          </Route>
          <Route path="export" element={<Export />} />
          <Route path="settings" element={<Settings />} />
          <Route path="feature-engineering">
            <Route index element={<FeatureEngineeringPage />} />
            <Route path="autoencoder" element={<FeatureEngineeringAutoencoderPage />} />
            <Route path="selection" element={<FeatureEngineeringSelectionPage />} />
            <Route path="extraction" element={<FeatureEngineeringExtractionPage />} />
          </Route>
          <Route path="transformation" element={<DataTransformationPage />} />
          <Route path="transformation/standardization" element={<StandardizationPage />} />
          <Route path="transformation/categorical" element={<CategoricalTransformationPage />} />
          <Route path="workflow" element={<WorkflowPage />} />
          <Route path="*" element={<Navigate to="" replace />} />
        </Routes>
      </div>
    </GalaxyLayout>
  );
}
