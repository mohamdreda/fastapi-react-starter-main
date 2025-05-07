import { Link, Outlet, useNavigate } from 'react-router-dom'
import { AppProvider } from '../context/AppContext'
import { Notification } from '../components/ui/Notification'
import { ThemeToggle } from '../components/ui/ThemeToggle'
import { useAuth } from '../context/AuthContext'
import { Button } from '../components/ui/Button'

const publicNavLinks = [
  { to: '/', label: 'Home' },
  { to: '/about', label: 'About' },
] as const

// Dashboard link will be dynamically generated based on user role

function Navigation() {
  const navigate = useNavigate()
  const { isAuthenticated, logout, user } = useAuth()
  const linkClasses =
    'text-gray-900 dark:text-white hover:text-gray-600 dark:hover:text-gray-300 px-3 py-2 rounded-md text-sm font-medium'

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <nav className="bg-white dark:bg-gray-800 shadow-sm transition-colors">
      <div className="container mx-auto px-6">
        <div className="flex justify-between h-16 items-center">
          <div className="flex space-x-8">
            {publicNavLinks.map(({ to, label }) => (
              <Link key={to} to={to} className={linkClasses}>
                {label}
              </Link>
            ))}
            {isAuthenticated && (
              <Link 
                to={user?.role === 'admin' ? '/admin/dashboard' : `/user/dashboard/${user?.id}`} 
                className={linkClasses}
              >
                Dashboard
              </Link>
            )}
          </div>
          <div className="flex items-center space-x-4">
            <ThemeToggle />
            {isAuthenticated ? (
              <Button
                variant="ghost"
                onClick={handleLogout}
                className="text-gray-900 dark:text-white hover:text-gray-600 dark:hover:text-gray-300"
              >
                Logout
              </Button>
            ) : (
              <Link to="/login" className={linkClasses}>
                Login
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}

export default function RootLayout() {
  const currentYear = new Date().getFullYear()

  return (
    <AppProvider>
      <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-gray-900 transition-colors">
        <Navigation />
        <main className="flex-1 container mx-auto px-6 py-8">
          <Outlet />
        </main>
        <footer className="bg-white dark:bg-gray-800 shadow-sm transition-colors mt-auto">
          <div className="container mx-auto px-6 py-4">
            <p className="text-center text-gray-500 dark:text-gray-400">
              {currentYear} FastAPI React Starter. All rights reserved.
            </p>
          </div>
        </footer>
        <Notification />
      </div>
    </AppProvider>
  )
}
