// app/chat/[sessionId]/page.tsx
'use client'

import { useState, useEffect, useRef, FC, ReactNode } from 'react';
import axios from 'axios';
import { useRouter } from 'next/navigation';
import { use } from 'react';
import ReactMarkdown from 'react-markdown';
import Sidebar from '@/components/Sidebar';
import DocumentInput from '@/components/DocumentInput';

// Message shape coming from the API
interface Message {
  id: string;
  content: string;
  is_user: boolean;
  created_at: string;
}

// Props for the dynamic route
interface ChatPageProps {
  params: Promise<{ sessionId: string }>;
}

// CodeBlock component for fenced code with copy button
interface CodeBlockProps {
  className?: string;
  children: ReactNode;
}

const CodeBlock: FC<CodeBlockProps> = ({ className, children }) => {
  const [copied, setCopied] = useState(false);
  const codeText = String(children).replace(/\n$/, '');

  const handleCopy = async () => {
    await navigator.clipboard.writeText(codeText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative group my-4">
      <pre className={`block overflow-auto w-full rounded-lg p-4 bg-gray-900 text-sm ${className ?? ''}`}>
        <code>{codeText}</code>
      </pre>
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-gray-700 hover:bg-gray-600 text-white text-xs px-2 py-1 rounded"
      >
        {copied ? 'Copied!' : 'Copy'}
      </button>
    </div>
  );
};

export default function ChatPage({ params }: ChatPageProps) {
  const { sessionId } = use(params);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [hasDocuments, setHasDocuments] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const router = useRouter();

  // useEffect(() => { 
  //   fetchMessages();
  //   checkIfSessionHasDocuments();
  // }, [sessionId]);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const fetchMessages = async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) return router.push('/login');
      const response = await axios.get(
        `/api/sessions/${sessionId}/messages`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setMessages(response.data.messages || []);
      if (response.data.messages && response.data.messages.length > 0) {
        setHasDocuments(true);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.error || 'Failed to fetch messages');
    }
  };

  // const checkIfSessionHasDocuments = async () => {
  //   try {
  //     const token = localStorage.getItem('token');
  //     if (!token) return;
      
  //     const sessionResponse = await axios.get(
  //       `/api/sessions/${sessionId}`,
  //       { headers: { Authorization: `Bearer ${token}` } }
  //     );
      
  //     if (sessionResponse.data && sessionResponse.data.doc_id) {
  //       setHasDocuments(true);
  //     } else if (sessionResponse.data && sessionResponse.data.linked_documents && sessionResponse.data.linked_documents.length > 0) {
  //       setHasDocuments(true);
  //     } else {
  //       setHasDocuments(false);
  //     }
      
  //   } catch (err) {
  //     console.error("Failed to check if session has documents:", err);
  //     setHasDocuments(false);
  //   }
  // };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = input.trim();
    setInput('');
    setIsLoading(true);
    setError('');

    try {
      const token = localStorage.getItem('token');
      if (!token) return router.push('/login');
      const response = await axios.post(
        `/api/sessions/${sessionId}/messages`,
        { question: userMessage },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      const user: Message = { id: Date.now().toString(), content: userMessage, is_user: true, created_at: new Date().toISOString() };
      const ai: Message = { id: (Date.now()+1).toString(), content: response.data.response, is_user: false, created_at: new Date().toISOString() };
      setMessages(prev => [...prev, user, ai]);

      // Dispatch session update event to refresh the sidebar
      window.dispatchEvent(new Event('sessionUpdate'));
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to send message');
    } finally { setIsLoading(false); }
  };

  return (
    <div className="flex h-full bg-[#1C1C1C] text-[#F5E8D8]">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <div className="p-4 border-b border-[#2A2A2A]">
          <div className="max-w-3xl mx-auto">
            <DocumentInput 
                sessionId={sessionId} 
                onDocumentProcessed={undefined}
                //onProcessingComplete={checkIfSessionHasDocuments}
            />
            {error && <div className="mt-2 p-3 bg-red-900/20 border border-red-700/30 rounded text-red-200">{error}</div>}
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          <div className="max-w-3xl mx-auto">
            {messages.length===0 && !hasDocuments ? (
              <div className="text-center py-8 text-[#F5E8D8]/60">
                <h2 className="text-2xl font-bold mb-4">Start a conversation</h2>
                <p>Process a documentation URL to begin.</p>
              </div>
            ) : (
              messages.map(msg => (
                <div key={msg.id} className={`mb-6 ${msg.is_user?'text-right':'text-left'}`}>
                  <div className={`inline-block max-w-[80%] p-4 rounded-lg ${
                    msg.is_user 
                      ? 'bg-[#FF6F61] text-[#F5E8D8] rounded-br-none' 
                      : 'bg-[#2A2A2A] text-[#F5E8D8] rounded-bl-none'
                  }`}>
                    {msg.is_user ? <div>{msg.content}</div> : (
                      <div className="prose prose-invert">
                        <ReactMarkdown
                          components={{
                            p({ node, children, ...props }) {
                              const first = (node as any).children[0];
                              if (first && first.type==='element' && first.tagName==='code') {
                                return <>{children}</>;
                              }
                              return <p {...props}>{children}</p>;
                            },
                            code({ inline, children, ...props }: any) {
                              const text = String(children).replace(/\n$/, '');
                              const wordCount = text.trim().split(/\s+/).length;
                              if (inline || wordCount < 3) {
                                return <code className="bg-[#1C1C1C] px-1 rounded" {...props}>{children}</code>;
                              }
                              return <CodeBlock {...props}>{children}</CodeBlock>;
                            }
                          }}>
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="mb-6 text-left">
                <div className="inline-block max-w-[80%] p-4 rounded-lg bg-[#2A2A2A] text-[#F5E8D8] rounded-bl-none animate-pulse">
                  AI is thinking...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
        <div className="p-4 border-t border-[#2A2A2A]">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSubmit} className="flex gap-2">
              <input 
                value={input} 
                onChange={e=>setInput(e.target.value)} 
                placeholder="Ask a question..." 
                className="flex-1 px-4 py-2 bg-[#2A2A2A] border border-[#3A3A3A] rounded text-[#F5E8D8] placeholder-[#F5E8D8]/40 focus:outline-none focus:ring-2 focus:ring-[#DAA520]"
              />
              <button 
                type="submit" 
                disabled={!input.trim()}
                className="bg-[#DAA520] hover:bg-[#DAA520]/90 text-[#1C1C1C] px-4 py-2 rounded disabled:opacity-50 transition-colors"
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
