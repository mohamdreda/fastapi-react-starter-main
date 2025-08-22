import { Link, Outlet, useNavigate, useLocation } from 'react-router-dom'
import { useEffect, useRef, useState } from 'react'
import { AppProvider } from '../context/AppContext'
import { Notification } from '../components/ui/Notification'
import { ThemeToggle } from '../components/ui/ThemeToggle'
import { useAuth } from '../context/AuthContext'
import Logo from '@/assets/data-cleaning-logo.svg'

const publicNavLinks = [
  { label: 'What is Data Cleaning?', id: 'what-is-data-cleaning' },
  { label: 'Tools', id: 'tools' },
  { label: 'About', id: 'about' },
  { label: 'Tech', id: 'tech' },
] as const

// Dashboard link will be dynamically generated based on user role

function Navigation() {
  const navigate = useNavigate()
  const { isAuthenticated, logout, user } = useAuth()
  const location = useLocation()
  const linkClasses =
    'text-gray-900 dark:text-white hover:text-gray-600 dark:hover:text-gray-300 px-3 py-2 rounded-md text-sm font-medium'
  const isHomePage = location.pathname === '/'

  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const initials = (
    ((user?.first_name?.[0] || '') + (user?.last_name?.[0] || '')).toUpperCase() ||
    (user?.email?.[0]?.toUpperCase() || 'U')
  )

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [])

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <nav className="bg-white dark:bg-gray-800 shadow-sm transition-colors">
      <div className="container mx-auto px-6">
        <div className="flex justify-between h-16 items-center">
          <div className="flex items-center space-x-8">
            <Link to="/" className="flex items-center space-x-2">
              <img src={Logo} alt="Data Cleaning" className="h-6 w-6" />
              <span className="font-semibold text-gray-900 dark:text-white">Data Cleaning</span>
            </Link>
            {isHomePage && publicNavLinks.map(({ id, label }) => (
              <button
                key={id}
                onClick={() => {
                  if (location.pathname !== '/') {
                    navigate('/')
                    setTimeout(() => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' }), 0)
                  } else {
                    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
                  }
                }}
                className={linkClasses}
              >
                {label}
              </button>
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
              <div className="relative" ref={menuRef}>
                <button
                  onClick={() => setMenuOpen((v) => !v)}
                  className="inline-flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium text-gray-900 dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  <span className="flex h-8 w-8 items-center justify-center rounded-full bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 text-xs font-semibold">
                    {initials}
                  </span>
                  <span className="hidden sm:inline">{user?.first_name || 'Profile'}</span>
                  <svg className="h-4 w-4 opacity-70" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.084l3.71-3.853a.75.75 0 111.08 1.04l-4.24 4.4a.75.75 0 01-1.08 0l-4.24-4.4a.75.75 0 01.02-1.06z" clipRule="evenodd" />
                  </svg>
                </button>
                {menuOpen && (
                  <div className="absolute right-0 mt-2 w-44 rounded-md border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-lg py-1 z-50">
                    <Link
                      to="/profile"
                      onClick={() => setMenuOpen(false)}
                      className="block w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700"
                    >
                      Profile
                    </Link>
                    <button
                      onClick={() => { setMenuOpen(false); handleLogout(); }}
                      className="block w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700"
                    >
                      Logout
                    </button>
                  </div>
                )}
              </div>
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
  const location = useLocation()
  const isAdminRoute = location.pathname.startsWith('/admin')
  const isUserDashboardRoute = location.pathname.startsWith('/user/dashboard')
  const isAppLayout = isAdminRoute || isUserDashboardRoute
  useEffect(() => {
    console.log('[RootLayout] mounted')
    return () => console.log('[RootLayout] unmounted')
  }, [])

  return (
    <AppProvider>
      <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900 transition-colors">
        {!isAppLayout && <Navigation />}
        <div className="flex-1 overflow-auto">
        <main className={`w-full h-full ${isAppLayout ? 'p-0' : 'p-6'}`}>
          <Outlet />
        </main>
      </div>
        {!isAppLayout && (
          <footer className="bg-white dark:bg-gray-800 shadow-sm transition-colors">
            <div className="container mx-auto px-6 py-4">
              <p className="text-center text-gray-500 dark:text-gray-400">
                {currentYear} Data Cleaning Platform. All rights reserved.
              </p>
            </div>
          </footer>
        )}
        <Notification />
      </div>
    </AppProvider>
  )
}
