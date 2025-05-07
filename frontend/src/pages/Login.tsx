import { Link } from 'react-router-dom'
import { useAuth } from '@/context/AuthContext'
import LoginForm from '@/features/auth/LoginForm'

export default function Login() {

  const { login } = useAuth()
  return (
    <div className="min-h-[80vh] flex flex-col items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="w-full max-w-md space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
            Sign in to your account
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600 dark:text-gray-400">
            Or{' '}
            <Link
              to="/register"
              className="font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300"
            >
              create a new account
            </Link>
          </p>
        </div>
        <LoginForm onSubmit={async (email: string, password: string) => {
          try {
            const result = await login({ email, password })
            if (!result.success) {
              throw new Error(result.error || 'Login failed')
            }
          } catch (error) {
            throw error
          }
        }} />
      </div>
    </div>
  )
}
