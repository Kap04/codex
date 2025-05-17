import { useState } from 'react';
import axios from 'axios';

interface DocumentInputProps {
  onDocumentProcessed: (docId: string) => void;
  sessionId: string;
}

export default function DocumentInput({ onDocumentProcessed, sessionId }: DocumentInputProps) {
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
      
      const response = await axios.post('/api/crawl', { url, session_id: sessionId }, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        timeout: 600000
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
    <div className="w-full max-w-2xl mx-auto">
      <form onSubmit={handleUrlSubmit} className="space-y-2">
        <label htmlFor="url" className="text-xs text-[#F5E8D8]/60 block">
          Documentation URL
        </label>
        <div className="flex gap-2">
          <input
            type="url"
            id="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com/docs"
            className="flex-1 px-3 py-1.5 bg-[#2A2A2A] border border-[#3A3A3A] rounded text-sm text-[#F5E8D8] placeholder-[#F5E8D8]/40 focus:outline-none focus:border-[#DAA520]"
            required
          />
          <button
            type="submit"
            disabled={isCrawling}
            className="p-2 bg-[#DAA520] text-[#1C1C1C] rounded hover:bg-[#DAA520]/90 disabled:bg-[#DAA520]/50 disabled:opacity-50 transition-colors flex items-center justify-center"
            aria-label="Process documentation"
          >
            {isCrawling ? (
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            ) : (
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            )}
          </button>
        </div>
      </form>
      
      {crawlStatus && (
        <div className="mt-2 text-xs text-[#DAA520]">
          {crawlStatus}
        </div>
      )}
      
      {error && (
        <div className="mt-2 text-xs text-[#FF6F61]">
          {error}
        </div>
      )}
    </div>
  );
} 