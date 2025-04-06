'use client'

import { useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import axios from 'axios';

export default function DocumentInputPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [url, setUrl] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const token = localStorage.getItem('token');
      if (!token) {
        router.push('/login');
        return;
      }

      const response = await axios.post('/api/crawl', { url }, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.data && response.data.doc_id) {
        // Redirect back to new chat page after successful document creation
        router.push('/chat/new');
      } else {
        setError('Failed to process document: Invalid response format');
      }
    } catch (error: any) {
      console.error('Failed to process document:', error);
      setError(error.response?.data?.error || 'Failed to process document');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
      <div className="max-w-md w-full mx-4">
        <h1 className="text-2xl font-bold mb-6 text-center">Process Documentation</h1>
        
        {searchParams.get('error') === 'no_documents' && (
          <div className="mb-4 p-4 bg-yellow-900/50 border border-yellow-700 rounded-md text-yellow-200">
            No documents found. Please process a documentation first to start a new chat.
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="url" className="block text-sm font-medium mb-2">
              Documentation URL
            </label>
            <input
              type="url"
              id="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/docs"
              className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600"
              required
            />
          </div>

          {error && (
            <div className="p-4 bg-red-900/50 border border-red-700 rounded-md text-red-200">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 rounded-md font-medium focus:outline-none focus:ring-2 focus:ring-blue-600 focus:ring-offset-2 focus:ring-offset-gray-900 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Processing...' : 'Process Documentation'}
          </button>
        </form>
      </div>
    </div>
  );
} 