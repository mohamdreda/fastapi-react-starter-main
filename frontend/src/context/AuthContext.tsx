import { createContext, useContext, useReducer, ReactNode, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { AuthState, LoginCredentials, RegisterCredentials, AuthResult } from '@/types/auth'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Action types
export const AUTH_ACTIONS = {
  LOGIN_SUCCESS: 'LOGIN_SUCCESS',
  LOGOUT: 'LOGOUT',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
} as const

type AuthAction =
  | { type: typeof AUTH_ACTIONS.LOGIN_SUCCESS; payload: { token: string; user: any } }
  | { type: typeof AUTH_ACTIONS.LOGOUT }
  | { type: typeof AUTH_ACTIONS.SET_LOADING; payload: boolean }
  | { type: typeof AUTH_ACTIONS.SET_ERROR; payload: string }

const AuthStateContext = createContext<AuthState | null>(null)
const AuthDispatchContext = createContext<{
  login: (credentials: LoginCredentials) => Promise<AuthResult>
  register: (credentials: RegisterCredentials) => Promise<AuthResult>
  logout: () => void
} | null>(null)

const getInitialState = (): AuthState => ({
  isAuthenticated: !!localStorage.getItem('token'),
  token: localStorage.getItem('token'),
  isLoading: false,
  error: null,
  user: JSON.parse(localStorage.getItem('user') || 'null'),
})

const authReducer = (state: AuthState, action: AuthAction): AuthState => {
  switch (action.type) {
    case AUTH_ACTIONS.LOGIN_SUCCESS:
      return {
        ...state,
        isAuthenticated: true,
        token: action.payload.token,
        user: action.payload.user,
        isLoading: false,
        error: null,
      }
    case AUTH_ACTIONS.LOGOUT:
      return {
        ...state,
        isAuthenticated: false,
        token: null,
        user: null,
        error: null,
      }
    case AUTH_ACTIONS.SET_LOADING:
      return { ...state, isLoading: action.payload }
    case AUTH_ACTIONS.SET_ERROR:
      return { ...state, error: action.payload, isLoading: false }
    default:
      return state
  }
}

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [state, dispatch] = useReducer(authReducer, getInitialState())
  const navigate = useNavigate()

  const login = useCallback(async (credentials: LoginCredentials): Promise<AuthResult> => {
    dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: true });
    
    try {
      // Convert credentials to form data format as expected by FastAPI
      const formData = new FormData();
      formData.append('email', credentials.email);
      formData.append('password', credentials.password);

      const response = await fetch(`${API_URL}/api/v1/auth/login`, {
        method: 'POST',
        body: formData,
      });
  
      const data = await response.json();
  
      if (!response.ok) {
        const errorMessage = data.detail || 'Login failed';
        dispatch({ type: AUTH_ACTIONS.SET_ERROR, payload: errorMessage });
        return { success: false, error: errorMessage };
      }
  
      const token = data.access_token;
      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify(data.user));
  
      dispatch({
        type: AUTH_ACTIONS.LOGIN_SUCCESS,
        payload: { token, user: data.user }
      });
  
      // Redirect based on role
      if (data.user.role === 'admin') {
        navigate('/admin/dashboard');
      } else {
        navigate(`/user/dashboard/${data.user.id}`);
      }
  
      return { success: true, error: null };
  
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Network error';
      dispatch({ type: AUTH_ACTIONS.SET_ERROR, payload: errorMessage });
      return { success: false, error: errorMessage };
    } finally {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: false });
    }
  }, [navigate]);
  
  const register = useCallback(async (credentials: RegisterCredentials): Promise<AuthResult> => {
    dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: true });
    
    try {
      const response = await fetch(`${API_URL}/api/v1/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: credentials.email,
          first_name: credentials.first_name,
          last_name: credentials.last_name,
          password: credentials.password,
          confirmPassword: credentials.confirmPassword
        }),
      });
  
      const data = await response.json();
  
      if (!response.ok) {
        let errorMessage = 'Registration failed';
        if (data.detail) {
          if (Array.isArray(data.detail)) {
            errorMessage = data.detail[0].msg;
          } else {
            errorMessage = data.detail;
          }
        }
        dispatch({ type: AUTH_ACTIONS.SET_ERROR, payload: errorMessage });
        return { success: false, error: errorMessage };
      }
  
      // Auto-login after successful registration
      return login({
        email: credentials.email,
        password: credentials.password
      });
    } catch (error) {
      dispatch({ type: AUTH_ACTIONS.SET_ERROR, payload: 'Network error' });
      return { success: false, error: 'Network error' };
    } finally {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: false });
    }
  }, [login]);

  const logout = useCallback(() => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    dispatch({ type: AUTH_ACTIONS.LOGOUT })
    navigate('/login')
  }, [navigate])

  // Effect to check token expiration
  useEffect(() => {
    if (state.token) {
      try {
        const payload = JSON.parse(atob(state.token.split('.')[1]));
        const expirationTime = payload.exp * 1000; // Convert to milliseconds
        
        if (Date.now() >= expirationTime) {
          logout();
        } else {
          // Set timeout to logout when token expires
          const timeout = setTimeout(() => {
            logout();
          }, expirationTime - Date.now());
          
          return () => clearTimeout(timeout);
        }
      } catch (error) {
        console.error('Error checking token expiration:', error);
        logout();
      }
    }
  }, [state.token, logout]);

  return (
    <AuthStateContext.Provider value={state}>
      <AuthDispatchContext.Provider value={{ login, register, logout }}>
        {children}
      </AuthDispatchContext.Provider>
    </AuthStateContext.Provider>
  );
}

export function useAuthState() {
  const context = useContext(AuthStateContext)
  if (!context) {
    throw new Error('useAuthState must be used within an AuthProvider')
  }
  return context
}

export function useAuthDispatch() {
  const context = useContext(AuthDispatchContext)
  if (!context) {
    throw new Error('useAuthDispatch must be used within an AuthProvider')
  }
  return context
}

export function useAuth() {
  return {
    ...useAuthState(),
    ...useAuthDispatch(),
  }
}
