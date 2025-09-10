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

      // Add timeout so fetch doesn't hang forever if API is unreachable
      const controller = new AbortController();
      const timeout = setTimeout(() => {
        controller.abort();
      }, 15000); // 15s timeout

      console.log('[Auth] Logging in to', `${API_URL}/api/v1/auth/login`);

      const response = await fetch(`${API_URL}/api/v1/auth/login`, {
        method: 'POST',
        body: formData,
        credentials: 'include',
        signal: controller.signal,
      });

      clearTimeout(timeout);
  
      let data: any = null;
      try {
        data = await response.json();
      } catch (e) {
        // If server didn't return JSON, provide a generic message
        console.warn('[Auth] Non-JSON response from /auth/login', e);
      }
  
      if (!response.ok) {
        const errorMessage = (data && data.detail) || `Login failed (${response.status})`;
        console.error('[Auth] Login failed:', errorMessage);
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
  
      // Auto-start a workflow session and store id (best-effort)
      try {
        const sessResp = await fetch(`${API_URL}/api/v1/sessions/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
          },
          body: JSON.stringify({
            title: `Session ${new Date().toISOString()}`,
            description: 'Auto-started on login',
          }),
        });
        if (sessResp.ok) {
          const sess = await sessResp.json();
          if (sess?.id) {
            localStorage.setItem('active_session_id', sess.id);
          }
        } else {
          console.warn('[Auth] Failed to auto-create session:', sessResp.status);
        }
      } catch (e) {
        console.warn('[Auth] Session creation skipped (non-blocking):', e);
      }

      // Redirect based on role
      if (data.user.role === 'admin') {
        navigate('/admin/dashboard');
      } else {
        navigate(`/user/dashboard/${data.user.id}`);
      }
  
      return { success: true, error: null };
  
    } catch (error: any) {
      const isAbort = error?.name === 'AbortError';
      const errorMessage = isAbort ? 'Login request timed out. Check API URL or backend availability.' : (error instanceof Error ? error.message : 'Network error');
      console.error('[Auth] Login error:', errorMessage);
      dispatch({ type: AUTH_ACTIONS.SET_ERROR, payload: errorMessage });
      return { success: false, error: errorMessage };
    } finally {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: false });
    }
  }, [navigate]);
  
  const register = useCallback(async (credentials: RegisterCredentials): Promise<AuthResult> => {
    dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: true });
    
    try {
      const formData = new FormData();
        formData.append('email', credentials.email);
        formData.append('first_name', credentials.first_name);
        formData.append('last_name', credentials.last_name);
        formData.append('password', credentials.password);

        const response = await fetch(`${API_URL}/api/v1/auth/register`, {
          method: 'POST',
          body: formData,
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
  
      // Do not auto-login; let user login manually after registration
      return { success: true, error: null };
    } catch (error) {
      dispatch({ type: AUTH_ACTIONS.SET_ERROR, payload: 'Network error' });
      return { success: false, error: 'Network error' };
    } finally {
      dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: false });
    }
  }, [login]);

  const logout = useCallback(() => {
    // Best-effort close current session
    const sid = localStorage.getItem('active_session_id')
    const token = localStorage.getItem('token')
    if (sid && token) {
      fetch(`${API_URL}/api/v1/sessions/${sid}/close`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
      }).catch(() => {})
    }

    localStorage.removeItem('token')
    localStorage.removeItem('user')
    localStorage.removeItem('active_session_id')
    dispatch({ type: AUTH_ACTIONS.LOGOUT })
    navigate('/login')
  }, [navigate])

  // Effect to check token expiration
  useEffect(() => {
    if (state.token) {
      try {
        const payload = JSON.parse(atob(state.token.split('.')[1]));
        let expirationTime = payload.exp * 1000; // Convert to milliseconds
        
        const schedule = () => {
          const now = Date.now();
          const refreshTime = expirationTime - 120_000; // refresh 2min before expiry
          if (now >= expirationTime) {
            logout();
            return;
          }
          const delay = Math.max(refreshTime - now, 0);
          const timeout = setTimeout(async () => {
            try {
              const resp = await fetch(`${API_URL}/api/v1/auth/refresh`, {
                method: 'POST',
                credentials: 'include',
              });
              if (resp.ok) {
                const data = await resp.json();
                localStorage.setItem('token', data.access_token);
                dispatch({
                  type: AUTH_ACTIONS.LOGIN_SUCCESS,
                  payload: { token: data.access_token, user: state.user },
                });
                // schedule next refresh
                const newPayload = JSON.parse(atob(data.access_token.split('.')[1]));
                expirationTime = newPayload.exp * 1000;
                schedule();
              } else {
                logout();
              }
            } catch {
              logout();
            }
          }, delay);
          return () => clearTimeout(timeout);
        };
        return schedule();
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
