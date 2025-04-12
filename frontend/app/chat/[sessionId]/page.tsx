// app/chat/[sessionId]/page.tsx
'use client'

import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { useRouter } from 'next/navigation';
import Markdown from 'react-markdown';
import Sidebar from '@/components/Sidebar';
import DocumentInput from '@/components/DocumentInput';
import { use } from 'react';

interface Message {
  id: string;
  content: string;
  is_user: boolean;
  created_at: string;
}

interface ChatPageProps {
  params: Promise<{
    sessionId: string;
  }>;
}

export default function ChatPage({ params }: ChatPageProps) {
  const { sessionId } = use(params);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
 // const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string>('');
  const [docId, setDocId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  useEffect(() => {
    fetchMessages();
  }, [sessionId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // const handleDeleteSession = async () => {
  //   try {
  //     setIsDeleting(true);
  //     setError('');

  //     const token = localStorage.getItem('token');
  //     if (!token) {
  //       router.push('/login');
  //       return;
  //     }

  //     const response = await axios.delete(`/api/sessions/${sessionId}`, {
  //       headers: {
  //         'Authorization': `Bearer ${token}`
  //       }
  //     });

  //     if (response.data.newSessionId) {
  //       router.push(`/chat/${response.data.newSessionId}`);
  //     }
  //   } catch (err: any) {
  //     console.error('Error deleting session:', err);
  //     setError(err.response?.data?.error || 'Failed to delete session');
  //   } finally {
  //     setIsDeleting(false);
  //   }
  // };

  const fetchMessages = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        router.push('/login');
        return;
      }

      const response = await axios.get(`/api/sessions/${sessionId}/messages`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.data && response.data.messages) {
        setMessages(response.data.messages);
      } else {
        setMessages([]);
      }
    } catch (err: any) {
      console.error('Error fetching messages:', err);
      setError(err.response?.data?.error || 'Failed to fetch messages');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    if (!docId) {
      setError('Please process a documentation first before asking questions.');
      return;
    }

    const userMessage = input.trim();
    setInput('');
    setIsLoading(true);
    setError('');

    try {
      const token = localStorage.getItem('token');
      if (!token) {
        router.push('/login');
        return;
      }

      const response = await axios.post(`/api/sessions/${sessionId}/messages`, {
        question: userMessage
      }, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      const userMessageObj = {
        id: Date.now().toString(),
        content: userMessage,
        is_user: true,
        created_at: new Date().toISOString()
      };
      
      const aiMessageObj = {
        id: (Date.now() + 1).toString(),
        content: response.data.response,
        is_user: false,
        created_at: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, userMessageObj, aiMessageObj]);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to send message');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDocumentProcessed = (newDocId: string) => {
    setDocId(newDocId);
    setError('');
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white">
      <Sidebar />
      
      <div className="flex-1 flex flex-col h-full">
        <div className="flex-1 overflow-y-auto p-4">
          <div className="max-w-3xl mx-auto">
            <div className="flex justify-end mb-4">
              {/* <button
                onClick={handleDeleteSession}
                disabled={isDeleting}
                className="text-gray-400 hover:text-red-500 disabled:opacity-50"
              >
                {isDeleting ? 'Deleting...' : 'Delete Chat'}
              </button> */}
            </div>
                
            {messages.length === 0 && !docId ? (
              <div className="text-center py-8 text-gray-400">
                <h2 className="text-2xl font-bold mb-4">Start a new conversation</h2>
                <p>Process a documentation URL to begin chatting.</p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`mb-6 ${message.is_user ? 'text-right' : 'text-left'}`}
                >
                  <div
                    className={`inline-block max-w-[80%] p-4 rounded-lg ${
                      message.is_user 
                        ? 'bg-blue-700 text-white rounded-br-none' 
                        : 'bg-gray-800 text-white rounded-bl-none'
                    }`}
                  >
                    {message.is_user ? (
                      <div>{message.content}</div>
                    ) : (
                      <div className="prose prose-invert">
                        <Markdown>{message.content}</Markdown>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="mb-6 text-left">
                <div className="inline-block max-w-[80%] p-4 rounded-lg bg-gray-800 text-white rounded-bl-none">
                  <div className="flex items-center gap-2">
                    <div className="animate-pulse">AI is thinking...</div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
        
        <div className="p-4 border-t border-gray-800">
          <div className="max-w-3xl mx-auto">
            <DocumentInput onDocumentProcessed={handleDocumentProcessed} />
            
            {error && (
              <div className="mb-4 p-3 bg-red-900/50 border border-red-700 rounded-md text-red-200">
                {error}
              </div>
            )}
            <form onSubmit={handleSubmit} className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question about the documentation..."
                className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600"
                disabled={isLoading || !docId}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim() || !docId}
                className="bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-blue-800 disabled:opacity-50"
              >
                Send
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}