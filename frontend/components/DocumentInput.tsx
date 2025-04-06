import { useState } from 'react';
import axios from 'axios';

interface DocumentInputProps {
  onDocumentProcessed: (docId: string) => void;
}

export default function DocumentInput({ onDocumentProcessed }: DocumentInputProps) {
  const [url, setUrl] = useState('');
  const [isCrawling, setIsCrawling] = useState(false);
  const [error, setError] = useState('');
  const [crawlStatus, setCrawlStatus] = useState('');

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsCrawling(true);
    setCrawlStatus('Starting crawl process...');
    
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        throw new Error('Not authenticated. Please log in.');
      }
      
      const response = await axios.post('/api/crawl', { url }, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      setCrawlStatus('Documentation successfully processed!');
      onDocumentProcessed(response.data.doc_id);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to process documentation');
      setCrawlStatus('');
    } finally {
      setIsCrawling(false);
    }
  };

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-md mb-8">
      <h2 className="text-xl font-semibold mb-4">Process Documentation</h2>
      <form onSubmit={handleUrlSubmit}>
        <div className="mb-4">
          <label htmlFor="url" className="block text-sm font-medium text-zinc-300 mb-1">
            Documentation URL
          </label>
          <input
            type="url"
            id="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/docs"
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-600"
            required
          />
        </div>
        <button
          type="submit"
          disabled={isCrawling}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-blue-800 disabled:opacity-50"
        >
          {isCrawling ? 'Processing...' : 'Process Documentation'}
        </button>
      </form>
      
      {crawlStatus && (
        <div className="mt-4 p-3 bg-blue-900/50 text-blue-200 rounded">
          {crawlStatus}
        </div>
      )}
      
      {error && (
        <div className="mt-4 p-3 bg-red-900/50 text-red-200 rounded">
          {error}
        </div>
      )}
    </div>
  );
} 