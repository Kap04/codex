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
      <div className="min-h-screen bg-[#1C1C1C] text-[#F5E8D8] flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#DAA520] mx-auto"></div>
          <p className="mt-4 text-[#F5E8D8]/80">Creating new chat session...</p>
        </div>
      </div>
    );
  }
  
  console.log('Rendering default state');
  return (
    <div className="flex h-screen bg-[#1C1C1C] text-[#F5E8D8]">
      <Sidebar />
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-xl font-semibold mb-4">Welcome to the Chat</h1>
          <p className="mt-4 text-[#F5E8D8]/80">You have existing sessions. Please select a session from the sidebar.</p>
        </div>
      </div>
    </div>
  );
} 