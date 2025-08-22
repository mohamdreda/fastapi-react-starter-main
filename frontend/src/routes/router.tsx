import React from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import RootLayout from '../layouts/RootLayout';
import AdminLayout from '../layouts/AdminLayout';
import { ProtectedRoute } from '../components/ProtectedRoute';
import { PageLoader } from '../components/ui/PageLoader';
import Login from '../pages/Login';
import Register from '../pages/Register';
import AdminItems from '../pages/AdminItems';
import AdminSettings from '../pages/AdminSettings';

// Lazy-loaded components
const Home = lazy(() => import('../pages/Home'));
const About = lazy(() => import('../pages/About'));
const AdminDashboard = lazy(() => import('../pages/AdminDashboard'));
const UserDashboard = lazy(() => import('../pages/UserDashboard'));
const Unauthorized = lazy(() => import('../pages/Unauthorized'));
const DeduplicationPage = lazy(() => import('../pages/DeduplicationPage'));
const ImputationPage = lazy(() => import('../pages/ImputationPage'));
const UserProfilePage = lazy(() => import('../pages/UserProfilePage'));
const WorkflowPage = lazy(() => import('../pages/Workflow'));

// Data Transformation Pages
const StandardizationPage = lazy(() => import('../pages/StandardizationPage'));
const CategoricalTransformationPage = lazy(() => import('../pages/CategoricalTransformationPage'));
const DataTransformationPage = lazy(() => import('../pages/DataTransformationPage'));

import ErrorBoundary from '../components/ErrorBoundary';

const withSuspense = (element: React.ReactNode) => (
  <Suspense fallback={<PageLoader />}>{element}</Suspense>
);

const routes = [
  {
    path: '/',
    element: <RootLayout />,
    errorElement: <ErrorBoundary />,
    children: [
      { index: true, element: withSuspense(<Home />) },
      { path: 'about', element: withSuspense(<About />) },
      { path: 'login', element: withSuspense(<Login />) },
      { path: 'register', element: withSuspense(<Register />) },
      { path: 'unauthorized', element: withSuspense(<Unauthorized />) },
      {
        path: 'profile',
        element: (
          <ProtectedRoute>
            {withSuspense(<UserProfilePage />)}
          </ProtectedRoute>
        ),
      },
      {
        path: 'admin',
        element: <ProtectedRoute role="admin" />,
        children: [
          {
            element: <AdminLayout />,
            children: [
              {
                index: true,
                element: withSuspense(<AdminDashboard />),
              },
              {
                path: 'dashboard',
                element: withSuspense(<AdminDashboard />),
              },
              {
                path: 'items',
                element: withSuspense(<AdminItems />),
              },
              {
                path: 'settings',
                element: withSuspense(<AdminSettings />),
              },
            ],
          },
        ],
      },
      {
        path: 'user',
        element: <ProtectedRoute role="user" />,
        children: [
          { 
            path: 'dashboard/:userId/*',
            element: withSuspense(<UserDashboard />),
            children: [
              // Autres routes utilisateur ici
              {
                path: 'clustering/density',
                element: withSuspense(
                  React.createElement(
                    React.lazy(() => import('../pages/ClusteringDensityPage'))
                  )
                )
              },
              {
                path: 'transformation/standardization',
                element: withSuspense(<StandardizationPage />)
              },
              {
                path: 'transformation/categorical',
                element: withSuspense(<CategoricalTransformationPage />)
              },
              {
                path: 'transformation',
                element: withSuspense(<DataTransformationPage />)
              },
              {
                path: 'duplicates',
                element: withSuspense(<DeduplicationPage />)
              },
              {
                path: 'imputation',
                element: withSuspense(<ImputationPage />)
              },
              {
                path: 'workflow',
                element: withSuspense(<WorkflowPage />)
              }
            ]
          }
        ]
      },
      // Redirect /dashboard to /user/dashboard
      {
        path: 'dashboard',
        element: <Navigate to="/user/dashboard" replace />
      },
      { path: '*', element: <Navigate to="/" replace /> }
    ]
  }
];

export const router = createBrowserRouter(routes);