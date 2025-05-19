import { useEffect } from 'react';
import { useNavigate, useParams, Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { GalaxyLayout } from '@/components/GalaxyLayout';
import { WelcomeDashboard } from '@/components/WelcomeDashboard';
import { DatasetManager } from '@/pages/DatasetManager';
import { DiagnosisDashboard } from '@/components/DataDiagnosis/DiagnosisDashboard';
<hr style="border: none; border-top: 4px double black;" />

// Placeholder components for other tools
const Imputation = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">Missing Values Imputation</h1>
    <p>Select a dataset to start handling missing values.</p>
  </div>
);
<hr style="border: none; border-top: 4px double black;" />

const Duplicates = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">Duplicates Detection</h1>
    <p>Select a dataset to find and remove duplicate records.</p>
  </div>
);

<hr style="border: none; border-top: 4px double black;" />


const Outliers = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">Outliers Detection</h1>
    <p>Select a dataset to identify and handle outliers.</p>
  </div>
);
<hr style="border: none; border-top: 4px double black;" />

const Export = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">Export Cleaned Data</h1>
    <p>Select a dataset to export after cleaning.</p>
  </div>
);
<hr style="border: none; border-top: 4px double black;" />

const Settings = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold mb-4">Settings</h1>
    <p>Configure your data cleaning parameters.</p>
  </div>
);

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
      <div className="min-h-screen flex flex-col"> {/* Make sure to use flex and min-h-screen */}
        <Routes>
          {/* Show welcome screen at root */}
          <Route index element={<WelcomeDashboard />} />

          <Route path="datasets" element={<DatasetManager />} />
          <Route path="diagnosis">
            <Route index element={<DiagnosisDashboard />} />
            <Route path=":datasetId" element={<DiagnosisDashboard />} />
            <Route path="outliers" element={<Outliers />} />
          </Route>
          <Route path="imputation" element={<Imputation />} />
          <Route path="duplicates" element={<Duplicates />} />
          <Route path="export" element={<Export />} />
          <Route path="settings" element={<Settings />} />

          {/* Redirect unknown paths to welcome screen */}
          <Route path="*" element={<Navigate to="" replace />} />
        </Routes>
      </div>
    </GalaxyLayout>
  );
}
