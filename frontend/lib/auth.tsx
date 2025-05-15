'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRouter } from 'next/navigation';

interface AuthContextType {
  isAuthenticated: boolean;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
  error: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    // Check for token in localStorage on mount
    const storedToken = localStorage.getItem('token');
    if (storedToken) {
      setToken(storedToken);
      setIsAuthenticated(true);
      
      // Also set token in cookie if it exists in localStorage but not in cookies
      if (!document.cookie.includes('token=')) {
        document.cookie = `token=${storedToken}; path=/; max-age=${60 * 60 * 24 * 7}`; // 7 days
      }
    } else {
      // Check for token in cookies
      const cookies = document.cookie.split(';');
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.startsWith('token=')) {
          const cookieToken = cookie.substring(6);
          setToken(cookieToken);
          setIsAuthenticated(true);
          localStorage.setItem('token', cookieToken);
          break;
        }
      }
    }
  }, []);

  const login = async (email: string, password: string) => {
    try {
      setError(null);
      console.log('Attempting login for:', email);
      
      const response = await fetch('http://localhost:5000/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();
      console.log('Login response:', data);

      if (!response.ok) {
        const errorMessage = data.error || 'Login failed';
        console.error('Login error:', errorMessage);
        setError(errorMessage);
        throw new Error(errorMessage);
      }

      setToken(data.token);
      setIsAuthenticated(true);
      localStorage.setItem('token', data.token);
      
      // Set token in cookie
      document.cookie = `token=${data.token}; path=/; max-age=${60 * 60 * 24 * 7}`; // 7 days
      
      // Check for existing sessions and create a new one if needed
      try {
        const sessionsResponse = await fetch('http://localhost:5000/sessions', {
          headers: {
            'Authorization': `Bearer ${data.token}`
          }
        });
        
        const sessionsData = await sessionsResponse.json();
        
        if (!sessionsResponse.ok) {
          console.error('Failed to fetch sessions:', sessionsData);
          throw new Error('Failed to fetch sessions');
        }
        
        if (!sessionsData.sessions || sessionsData.sessions.length === 0) {
          // Create a new session if none exist
          const newSessionResponse = await fetch('http://localhost:5000/sessions', {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${data.token}`,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              title: "New Chat",
              create_without_doc: true // Flag to indicate we want to create a session without requiring a document
            })
          });
          
          const newSessionData = await newSessionResponse.json();
          
          if (!newSessionResponse.ok) {
            console.error('Failed to create new session:', newSessionData);
            throw new Error('Failed to create new session');
          }
          
          if (newSessionData.session_id) {
            router.push(`/chat/${newSessionData.session_id}`);
            return;
          }
        }
      } catch (sessionError) {
        console.error('Error handling sessions:', sessionError);
      }
      
      router.push('/chat/new'); // Redirect to new chat after login
    } catch (error) {
      console.error('Login error:', error);
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
      throw error;
    }
  };

  const register = async (email: string, password: string) => {
    try {
      setError(null);
      console.log('Attempting registration for:', email);
      
      const response = await fetch('http://localhost:5000/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();
      console.log('Registration response:', data);

      if (!response.ok) {
        const errorMessage = data.error || 'Registration failed';
        console.error('Registration error:', errorMessage);
        setError(errorMessage);
        throw new Error(errorMessage);
      }

      setToken(data.token);
      setIsAuthenticated(true);
      localStorage.setItem('token', data.token);
      
      // Set token in cookie
      document.cookie = `token=${data.token}; path=/; max-age=${60 * 60 * 24 * 7}`; // 7 days
      
      // Create a new session for the new user
      try {
        const newSessionResponse = await fetch('http://localhost:5000/sessions', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${data.token}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            title: "New Chat",
            create_without_doc: true // Flag to indicate we want to create a session without requiring a document
          })
        });
        
        const newSessionData = await newSessionResponse.json();
        
        if (!newSessionResponse.ok) {
          console.error('Failed to create new session:', newSessionData);
          throw new Error('Failed to create new session');
        }
        
        if (newSessionData.session_id) {
          router.push(`/chat/${newSessionData.session_id}`);
          return;
        }
      } catch (sessionError) {
        console.error('Error creating new session:', sessionError);
      }
      
      router.push('/chat/new'); // Redirect to new chat after registration
    } catch (error) {
      console.error('Registration error:', error);
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
      throw error;
    }
  };

  const logout = () => {
    setToken(null);
    setIsAuthenticated(false);
    setError(null);
    localStorage.removeItem('token');
    
    // Remove token from cookie
    document.cookie = 'token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';
    
    router.push('/login');
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, token, login, register, logout, error }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
} 