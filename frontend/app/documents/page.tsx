'use client'

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';
import Sidebar from '@/components/Sidebar';

interface Document {
  id: string;
  doc_id: string;
  url: string;
  content: string;
  created_at: string;
}

export default function DocumentsPage() {
  const router = useRouter();
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          router.push('/login');
          return;
        }

        const response = await axios.get('/api/documents', {
          headers: {
            'Authorization': `Bearer ${token}`
          },
          timeout: 60000000 // 10 minutes
        });

        setDocuments(response.data);
      } catch (err: any) {
        setError(err.response?.data?.error || 'Failed to fetch documents');
      } finally {
        setLoading(false);
      }
    };

    fetchDocuments();
  }, [router]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const truncateContent = (content: string, maxLength: number = 200) => {
    if (content.length <= maxLength) return content;
    return content.substring(0, maxLength) + '...';
  };

  return (
    <div className="flex h-screen bg-[#1C1C1C] text-[#F5E8D8]">
      <Sidebar />
      
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-2xl font-bold mb-6">Processed Documents</h1>
          
          {loading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#DAA520] mx-auto"></div>
              <p className="mt-4 text-[#F5E8D8]/60">Loading documents...</p>
            </div>
          ) : error ? (
            <div className="p-4 bg-red-900/20 border border-red-700/30 rounded-md text-[#FF6F61]">
              {error}
            </div>
          ) : documents.length === 0 ? (
            <div className="text-center py-8 text-[#F5E8D8]/60">
              <p>No documents have been processed yet.</p>
              <button
                onClick={() => router.push('/chat/new/document')}
                className="mt-4 bg-[#FF6F61] hover:bg-[#FF6F61]/90 text-[#1C1C1C] py-2 px-4 rounded-md transition-colors"
              >
                Process New Document
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  className="bg-[#2A2A2A] rounded-lg p-6 hover:bg-[#2A2A2A]/80 transition-colors border border-[#3A3A3A]"
                >
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h2 className="text-lg font-semibold mb-2">
                        {doc.url}
                      </h2>
                      <p className="text-sm text-[#F5E8D8]/60">
                        Processed on {formatDate(doc.created_at)}
                      </p>
                    </div>
                    <button
                      onClick={() => router.push(`/chat/new?doc_id=${doc.doc_id}`)}
                      className="bg-[#DAA520] hover:bg-[#DAA520]/90 text-[#1C1C1C] py-2 px-4 rounded-md text-sm transition-colors"
                    >
                      Start Chat
                    </button>
                  </div>
                  <div className="text-[#F5E8D8]/80 text-sm">
                    <p className="whitespace-pre-wrap">
                      {truncateContent(doc.content)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 