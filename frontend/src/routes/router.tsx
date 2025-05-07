import { createBrowserRouter, Navigate } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import RootLayout from '../layouts/RootLayout';
import { ProtectedRoute } from '../components/ProtectedRoute';
import { PageLoader } from '../components/ui/PageLoader';

// Lazy-loaded components
const Home = lazy(() => import('../pages/Home'));
const About = lazy(() => import('../pages/About'));
const AdminDashboard = lazy(() => import('../pages/AdminDashboard'));
const UserDashboard = lazy(() => import('../pages/UserDashboard'));
const Login = lazy(() => import('../pages/Login'));
const Register = lazy(() => import('../pages/Register'));
const Unauthorized = lazy(() => import('../pages/Unauthorized'));

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
        path: 'admin',
        element: <ProtectedRoute role="admin" />,
        children: [
          { 
            path: 'dashboard',
            element: withSuspense(<AdminDashboard />) 
          }
        ]
      },
      {
        path: 'user',
        element: <ProtectedRoute role="user" />,
        children: [
          { 
            path: 'dashboard/:userId/*',
            element: withSuspense(<UserDashboard />),
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