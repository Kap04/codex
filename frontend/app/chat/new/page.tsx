'use client'

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';

export default function NewChatPage() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const createNewSession = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          console.log('No token found, redirecting to login');
          router.push('/login');
          return;
        }

        console.log('Creating new session with token');
        const response = await axios.post('/api/sessions', {}, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        console.log('Session created successfully:', response.data);
        if (response.data && response.data.session_id) {
          router.push(`/chat/${response.data.session_id}`);
        } else {
          console.error('No session_id in response:', response.data);
          setError('Failed to create session: Invalid response format');
        }
      } catch (error: any) {
        console.error('Failed to create new session:', error);
        if (error.response?.data?.code === 'NO_DOCUMENTS') {
          // Redirect to document input page if no documents exist
          router.push('/chat/new/document?error=no_documents');
        } else {
          setError(error.response?.data?.error || 'Failed to create new session');
        }
      }
    };

    createNewSession();
  }, [router]);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-2xl font-bold mb-4">Creating new chat session...</h1>
        {error ? (
          <div className="text-red-500 mb-4">{error}</div>
        ) : (
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto"></div>
        )}
      </div>
    </div>
  );
} 