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
    <div className="h-full w-56 bg-[#1C1C1C] text-[#F5E8D8] p-3 flex flex-col border-r border-[#2A2A2A]">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold">Codex</h2>
      </div>
      
      <button
        onClick={createNewSession}
        className="flex items-center justify-center gap-2 w-full bg-[#FF6F61] hover:bg-[#FF6F61]/90 text-[#F5E8D8] py-1.5 px-3 rounded-md mb-4 text-sm transition-colors"
      >
        <PlusCircle size={14} />
        <span>New Chat</span>
      </button>
      
      <div className="flex flex-col gap-0.5 flex-grow overflow-y-auto">
        <h3 className="text-xs text-[#F5E8D8]/60 font-medium px-2 mb-1">Recent Chats</h3>
        
        {loading ? (
          <div className="text-center py-2 text-[#F5E8D8]/60 text-sm">Loading...</div>
        ) : error ? (
          <div className="text-center py-2 text-red-400 text-sm">{error}</div>
        ) : sessions.length === 0 ? (
          <div className="text-center py-2 text-[#F5E8D8]/60 text-sm">No chats yet</div>
        ) : (
          sessions.map((session) => (
            <Link 
              href={`/chat/${session.id}`} 
              key={session.id}
              className={`flex items-center justify-between px-2 py-1.5 rounded-md hover:bg-[#2A2A2A] text-sm transition-colors ${
                pathname === `/chat/${session.id}` ? 'bg-[#2A2A2A]' : ''
              }`}
            >
              <div className="flex items-center gap-2 truncate">
                <MessageSquare size={14} className="text-[#F5E8D8]/60" />
                <span className="truncate">{session.title}</span>
              </div>
              <button
                onClick={(e) => deleteSession(session.id, e)}
                className="text-[#F5E8D8]/60 hover:text-[#FF6F61] p-0.5 transition-colors"
              >
                <Trash2 size={12} />
              </button>
            </Link>
          ))
        )}
      </div>
      
      <div className="mt-3 pt-3 border-t border-[#2A2A2A]">
        <Link 
          href="/" 
          className="flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-[#2A2A2A] text-sm transition-colors"
        >
          Home
        </Link>
      </div>
    </div>
  );
}