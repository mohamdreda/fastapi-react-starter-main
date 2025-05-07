export interface User {
  id: string
  email: string
  role: 'user' | 'admin' | 'moderator'
}

export interface LoginCredentials {
  email: string
  password: string
}

export interface RegisterCredentials extends LoginCredentials {
  first_name: string
  last_name: string
  confirmPassword?: string
}

export interface AuthResponse {
  access_token: string
  token_type: string
}

export interface AuthResult {
  success: boolean
  error: string | null
}

export interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
}
