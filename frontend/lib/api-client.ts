import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from 'axios';

// Create a custom instance of axios
const apiClient = axios.create({
  baseURL: '/',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Function to refresh the token
async function refreshAuthToken(refreshTokenStr: string): Promise<string | null> {
  try {
    const response = await axios.post('/api/auth/refresh', { refresh_token: refreshTokenStr });
    if (response.data.access_token) {
      localStorage.setItem('token', response.data.access_token);
      localStorage.setItem('refreshToken', response.data.refresh_token);
      localStorage.setItem('tokenExpiry', response.data.expires_at);
      return response.data.access_token;
    }
    return null;
  } catch (error) {
    console.error('Error refreshing token:', error);
    return null;
  }
}

// Add a request interceptor
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };
    
    // If the error is a 401 (unauthorized) and we haven't retried yet
    if (error.response?.status === 401 && !originalRequest._retry && 
        error.response?.data && (error.response.data as any).code === 'TOKEN_EXPIRED') {
      
      originalRequest._retry = true;
      const refreshTokenValue = localStorage.getItem('refreshToken');
      
      if (refreshTokenValue) {
        const newToken = await refreshAuthToken(refreshTokenValue);
        
        if (newToken) {
          // Retry the original request with the new token
          if (originalRequest.headers) {
            originalRequest.headers['Authorization'] = `Bearer ${newToken}`;
          } else {
            originalRequest.headers = { 'Authorization': `Bearer ${newToken}` };
          }
          
          return apiClient(originalRequest);
        }
      }
      
      // If we couldn't refresh the token, redirect to login
      localStorage.removeItem('token');
      localStorage.removeItem('refreshToken');
      localStorage.removeItem('tokenExpiry');
      window.location.href = '/login';
    }
    
    return Promise.reject(error);
  }
);

export default apiClient; 