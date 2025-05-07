import { Navigate, useLocation, Outlet } from 'react-router-dom'
import { useAuth } from '@/context/AuthContext'
import { PageLoader } from './ui/PageLoader'

interface ProtectedRouteProps {
  children?: React.ReactNode
  role?: 'admin' | 'user'
}

export function ProtectedRoute({ children, role }: ProtectedRouteProps) {
  const { isAuthenticated, isLoading, user } = useAuth()
  const location = useLocation()

  if (isLoading) return <PageLoader />

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  if (role && user?.role !== role) {
    return <Navigate to="/" state={{ message: 'Unauthorized access' }} replace />
  }

  return children ? <>{children}</> : <Outlet />
}