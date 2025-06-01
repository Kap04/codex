import { useState } from 'react';
import axios from 'axios';

interface DocumentInputProps {
  onDocumentProcessed?: (docId: string) => void;
  sessionId: string;
  onProcessingComplete?: () => void;
}

export default function DocumentInput({ onDocumentProcessed, sessionId, onProcessingComplete }: DocumentInputProps) {
  const [primaryUrl, setPrimaryUrl] = useState<string>('');
  const [listUrls, setListUrls] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<Record<string, { status: string, error: string }>>({});

  const handleAddUrlInput = () => {
    if (primaryUrl.trim()) {
      setListUrls([primaryUrl.trim(), ...listUrls]);
      setPrimaryUrl('');
      setProcessingStatus(prev => {
        const newStatus = { ...prev };
        delete newStatus[primaryUrl.trim()];
        return newStatus;
      });
    }
  };

  const handleRemoveListUrl = (index: number) => {
    const newUrls = listUrls.filter((_, i) => i !== index);
    setListUrls(newUrls);
    setProcessingStatus(prev => {
      const removedUrl = listUrls[index];
      const newStatus = { ...prev };
      delete newStatus[removedUrl];
      return newStatus;
    });
  };

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const allUrlsToProcess = [...listUrls, primaryUrl.trim()].filter(url => url !== '');

    if (allUrlsToProcess.length === 0) {
      return;
    }

    setIsProcessing(true);
    setProcessingStatus({});

    const token = localStorage.getItem('token');
    if (!token) {
      setProcessingStatus(allUrlsToProcess.reduce((acc, url) => ({ ...acc, [url]: { status: '', error: 'Not authenticated. Please log in.' } }), {}));
      setIsProcessing(false);
      return;
    }

    const results = await Promise.all(allUrlsToProcess.map(async (url) => {
      setProcessingStatus(prev => ({ ...prev, [url]: { status: `Crawling ${url}...`, error: '' } }));
      try {
        const response = await axios.post('/api/crawl', { url: url.trim(), session_id: sessionId }, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          timeout: 600000
        });

        setProcessingStatus(prev => ({ ...prev, [url]: { status: `Successfully processed ${url}!`, error: '' } }));
        return { url, success: true, docId: response.data.doc_id };
      } catch (err: any) {
        const errorMsg = err.response?.data?.error || `Failed to process ${url}`;
        setProcessingStatus(prev => ({ ...prev, [url]: { status: '', error: errorMsg } }));
        return { url, success: false, error: errorMsg };
      }
    }));

    setIsProcessing(false);

    if (onProcessingComplete) {
      onProcessingComplete();
    }
  };

  const allUrlsForStatusDisplay = Array.from(new Set([...listUrls, primaryUrl].filter(url => url.trim() !== '')));

  return (
    <div className="w-full max-w-2xl mx-auto">
      <form onSubmit={handleUrlSubmit} className="space-y-2">
        <label htmlFor="primary-url" className="text-xs text-[#F5E8D8]/60 block">
          Documentation URL
        </label>
        <div className="flex gap-2">
          <input
            type="url"
            id="primary-url"
            value={primaryUrl}
            onChange={(e) => setPrimaryUrl(e.target.value)}
            placeholder="https://example.com/docs"
            className={`flex-1 px-3 py-1.5 bg-[#2A2A2A] border rounded text-sm text-[#F5E8D8] placeholder-[#F5E8D8]/40 focus:outline-none border-[#3A3A3A] focus:border-[#DAA520]`}
            required={listUrls.length === 0}
          />
          <button
            type="button"
            onClick={handleAddUrlInput}
            className="p-2 bg-[#3A3A3A] text-[#F5E8D8] rounded hover:bg-[#4A4A4A] transition-colors flex items-center justify-center"
            aria-label="Add another URL input"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
          </button>
          <button
            type="submit"
            disabled={isProcessing || ([...listUrls, primaryUrl].filter(url => url.trim() !== '').length === 0)}
            className="p-2 bg-[#DAA520] text-[#1C1C1C] rounded hover:bg-[#DAA520]/90 disabled:bg-[#DAA520]/50 disabled:opacity-50 transition-colors flex items-center justify-center"
            aria-label="Process documentation"
          >
            {isProcessing ? (
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

      {listUrls.length > 0 && (
        <div className="mt-2 space-y-1">
          <label className="text-xs text-[#F5E8D8]/60 block">URLs to process:</label>
          {listUrls.map((url, index) => (
            <div key={index} className="flex gap-2 items-center text-sm text-[#F5E8D8]">
              <div className="flex-1 overflow-hidden text-ellipsis whitespace-nowrap">
                {url}
              </div>
              <button
                type="button"
                onClick={() => handleRemoveListUrl(index)}
                className="p-1 text-[#F5E8D8] rounded hover:text-[#FF6F61] transition-colors flex items-center justify-center"
                aria-label="Remove URL"
              >
                <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="mt-4 space-y-2">
        {allUrlsForStatusDisplay.map((url) => {
          const statusInfo = processingStatus[url];
          if (!statusInfo) return null;
          return (
            <div key={url} className={`text-xs ${statusInfo.error ? 'text-[#FF6F61]' : 'text-[#DAA520]'}`}>
              {statusInfo.status || statusInfo.error}
            </div>
          );
        })}
      </div>
    </div>
  );
} 