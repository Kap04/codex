// app/page.tsx
'use client'

import { useState } from 'react';
import axios from 'axios';
import Markdown from 'react-markdown'

export default function Home() {
  const [url, setUrl] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  //const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCrawling, setIsCrawling] = useState(false);
  const [isAnswering, setIsAnswering] = useState(false);
  const [error, setError] = useState('');
  const [crawlStatus, setCrawlStatus] = useState('');

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsCrawling(true);
    setCrawlStatus('Starting crawl process...');
    
    try {
      const response = await axios.post('../api/crawl', { url });
      setCrawlStatus('Documentation successfully processed!');
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to process documentation');
      setCrawlStatus('');
    } finally {
      setIsCrawling(false);
    }
  };

  const handleQuestionSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsAnswering(true);
    
    try {
      const response = await axios.post('/api/ask', { question });
      setAnswer(response.data.answer);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to get answer');
    } finally {
      setIsAnswering(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <main className="max-w-3xl  mx-auto">
        <h1 className="text-3xl font-bold mb-8 text-center">Codex</h1>
        <p className="mb-6 text-gray">
          Enter a documentation URL to process, then ask questions about it.
        </p>
        
        <div className="bg-gray-900 p-6 rounded-lg shadow-md mb-8">
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
                className="w-full px-3 py-2 border text-white border-gray-300 rounded-md"
                required
              />
            </div>
            <button
              type="submit"
              disabled={isCrawling}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-blue-400"
            >
              {isCrawling ? 'Processing...' : 'Process Documentation'}
            </button>
          </form>
          
          {crawlStatus && (
            <div className="mt-4 p-3 bg-blue-50 text-blue-800 rounded">
              {crawlStatus}
            </div>
          )}
        </div>
        
        {crawlStatus.includes('successfully') && (
          <div className="bg-gray-900 p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Ask Questions</h2>
            <form onSubmit={handleQuestionSubmit}>
              <div className="mb-4">
                <label htmlFor="question" className="block text-sm font-medium text-zinc-300 mb-1">
                  Your Question
                </label>
                <textarea
                  id="question"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="What does this documentation say about...?"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  rows={3}
                  required
                />
              </div>
              <button
                type="submit"
                disabled={isAnswering}
                className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 disabled:bg-green-400"
              >
                {isAnswering ? 'Getting Answer...' : 'Get Answer'}
              </button>
            </form>
            
            {answer && (
              <div className="mt-6">
                <h3 className="font-medium text-lg mb-2">Answer:</h3>
                <div className="bg-gray-900 p-4 rounded-md whitespace-pre-wrap">
                  <Markdown>{answer}</Markdown>
                </div>
              </div>
            )}
          </div>
        )}
        
        {error && (
          <div className="mt-6 p-4 bg-red-50 text-red-700 rounded-md">
            <strong>Error:</strong> {error}
          </div>
        )}
      </main>
    </div>
  );
}