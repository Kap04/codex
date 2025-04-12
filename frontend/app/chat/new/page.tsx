'use client'

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';
import Sidebar from '@/components/Sidebar';

export default function NewChatPage() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const createNewSession = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          console.log('No token found, redirecting to login');
          router.push('/login');
          return;
        }

        console.log('Checking existing sessions...');
        const sessionsResponse = await axios.get('/api/sessions', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        console.log('Sessions response:', sessionsResponse.data);

        if (sessionsResponse.data.sessions && sessionsResponse.data.sessions.length === 0) {
          console.log('No existing sessions, creating a new session...');
          const response = await axios.post('/api/sessions', {}, {
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json'
            }
          });

          console.log('Session creation response:', response.data);
          if (response.data && response.data.session_id) {
            router.push(`/chat/${response.data.session_id}`);
          } else {
            console.error('No session_id in response:', response.data);
            setError('Failed to create session: Invalid response format');
          }
        } else {
          console.log('Existing sessions found, not creating a new session.');
        }
      } catch (error: any) {
        console.error('Failed to create new session:', error);
        if (error.response?.status === 401) {
          console.log('Token expired or invalid, redirecting to login');
          localStorage.removeItem('token'); // Clear invalid token
          router.push('/login');
          return;
        }
        if (error.response?.data?.code === 'NO_DOCUMENTS') {
          router.push('/chat/new/document?error=no_documents');
        } else {
          setError(error.response?.data?.error || 'Failed to create new session');
        }
      } finally {
        console.log('Setting loading to false');
        setLoading(false);
      }
    };

    createNewSession();
  }, [router]);

  if (loading) {
    console.log('Rendering loading state');
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4">Creating new chat session...</p>
        </div>
      </div>
    );
  }

  if (error) {
    console.log('Rendering error state');
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <Sidebar />
        <div className="max-w-md w-full mx-4 p-6 bg-gray-800 rounded-lg shadow-lg">
          <h1 className="text-xl font-semibold mb-4">Error</h1>
          <p className="text-red-400 mb-4">{error}</p>
          <button
            onClick={() => router.push('/chat/new/document')}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  console.log('Rendering default state');
  return (
    <div className="flex h-screen bg-gray-900 text-white">
      <Sidebar />
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-xl font-semibold mb-4">Welcome to the Chat</h1>
          <p className="mt-4">You have existing sessions. Please select a session from the sidebar.</p>
        </div>
      </div>
    </div>
  );
} 