// app/api/crawl/route.ts
import { NextRequest, NextResponse } from 'next/server';
import axios, { AxiosError, AxiosRequestConfig } from 'axios';
import https from 'https';
import http from 'http';

import axiosRetry from 'axios-retry';



// Configure axios-retry for automatic retries
axiosRetry(axios, {
  retries: 3,
  retryDelay: axiosRetry.exponentialDelay,
  retryCondition: (error: AxiosError) => {
    // Retry on connection reset, timeout, or specific server errors (500, 502, 503, 504)
    return (
      error.code === 'ECONNRESET' ||
      error.code === 'ETIMEDOUT' ||
      (error.response !== undefined &&
        [500, 502, 503, 504].includes(error.response.status))
    );
  }
});

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { url } = body;
    
    if (!url) {
      return NextResponse.json({ error: 'URL is required' }, { status: 400 });
    }
    
    // Log the URL being sent
    console.log('Sending URL to backend:', url);
    
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:5000';
    
    // Axios request configuration with increased timeout (5 minutes)
    const axiosConfig: AxiosRequestConfig = {
      timeout: 300000, // 5 minutes
      httpAgent: new http.Agent({ keepAlive: true }),
      httpsAgent: new https.Agent({ keepAlive: true }),
      headers: {
        'Content-Type': 'application/json',
        'Connection': 'keep-alive'
      }
    };

    try {
      const response = await axios.post(`${backendUrl}/crawl`, { url }, axiosConfig);
      
      return NextResponse.json(response.data);
    } catch (error) {
      // Type-safe error handling
      if (axios.isAxiosError(error)) {
        console.error('Detailed Axios Error:', {
          message: error.message,
          code: error.code,
          response: error.response?.data,
          status: error.response?.status,
          stack: error.stack
        });
        
        // Specific error handling
        if (error.code === 'ECONNRESET') {
          return NextResponse.json({ 
            error: 'Connection was reset. The server might be overloaded or the URL may be taking too long to process.',
            details: error.message
          }, { status: 500 });
        }
        
        return NextResponse.json({ 
          error: 'Failed to process documentation',
          details: error.response?.data || error.message
        }, { status: 500 });
      }
      
      // Handle non-Axios errors
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error('Unexpected error:', error);
      return NextResponse.json({ 
        error: 'Unexpected error occurred',
        details: errorMessage
      }, { status: 500 });
    }
  } catch (error) {
    // Handle request body parsing error
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error('Unexpected error:', error);
    return NextResponse.json({ 
      error: 'Unexpected error occurred',
      details: errorMessage
    }, { status: 500 });
  }
}