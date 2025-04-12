// components/Sidebar.tsx
'use client'

import { useState, useEffect } from 'react';
import axios from 'axios';
import Link from 'next/link';
import { PlusCircle, Trash2, MessageSquare } from 'lucide-react';
import { useRouter, usePathname } from 'next/navigation';

interface Session {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export default function Sidebar() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      router.push('/login');
      return;
    }

    const fetchSessions = async () => {
      try {
        const response = await axios.get('/api/sessions', {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        setSessions(response.data.sessions);
      } catch (err: any) {
        setError(err.response?.data?.error || 'Failed to fetch sessions');
      } finally {
        setLoading(false);
      }
    };

    fetchSessions();
  }, [router]);

  const createNewSession = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post('/api/sessions', {}, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      
      router.push(`/chat/${response.data.session_id}`);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to create new session');
    }
  };

  const deleteSession = async (id: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (!confirm('Are you sure you want to delete this chat?')) return;
    
    try {
      const token = localStorage.getItem('token');
      await axios.delete(`/api/sessions/${id}`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      
      setSessions(sessions.filter(session => session.id !== id));
      
      // If we're currently viewing this session, create a new session and redirect
      if (pathname === `/chat/${id}`) {
        const response = await axios.post('/api/sessions', {}, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        router.push(`/chat/${response.data.session_id}`);
      }
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to delete session');
    }
  };

  return (
    <div className="h-full w-64 bg-gray-800 text-white p-4 flex flex-col">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold">Codex</h2>
      </div>
      
      <button
        onClick={createNewSession}
        className="flex items-center justify-center gap-2 w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-md mb-6"
      >
        <PlusCircle size={16} />
        <span>New Chat</span>
      </button>
      
      <div className="flex flex-col gap-1 flex-grow overflow-y-auto">
        <h3 className="text-sm text-gray-400 font-medium px-2 mb-2">Recent Chats</h3>
        
        {loading ? (
          <div className="text-center py-4 text-gray-400">Loading...</div>
        ) : error ? (
          <div className="text-center py-4 text-red-400">{error}</div>
        ) : sessions.length === 0 ? (
          <div className="text-center py-4 text-gray-400">No chats yet</div>
        ) : (
          sessions.map((session) => (
            <Link 
              href={`/chat/${session.id}`} 
              key={session.id}
              className={`flex items-center justify-between px-3 py-2 rounded-md hover:bg-gray-700 ${
                pathname === `/chat/${session.id}` ? 'bg-gray-700' : ''
              }`}
            >
              <div className="flex items-center gap-2 truncate">
                <MessageSquare size={16} className="text-gray-400" />
                <span className="truncate">{session.title}</span>
              </div>
              <button
                onClick={(e) => deleteSession(session.id, e)}
                className="text-gray-400 hover:text-red-400 p-1"
              >
                <Trash2 size={14} />
              </button>
            </Link>
          ))
        )}
      </div>
      
      <div className="mt-4 pt-4 border-t border-gray-700">
        <Link 
          href="/" 
          className="flex items-center gap-2 px-3 py-2 rounded-md hover:bg-gray-700"
        >
          Home
        </Link>
      </div>
    </div>
  );
}