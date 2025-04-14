import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

export async function POST(request: NextRequest) {
  try {
    const { refresh_token } = await request.json();
    
    if (!refresh_token) {
      return NextResponse.json({ error: 'Refresh token is required' }, { status: 400 });
    }

    const baseUrl = process.env.BACKEND_URL || 'http://127.0.0.1:5000';
    const response = await axios.post(`${baseUrl}/auth/refresh`, { refresh_token });

    return NextResponse.json(response.data);
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Token refresh error:', error.response?.data);
      return NextResponse.json({ 
        error: 'Failed to refresh token',
        details: error.response?.data || error.message
      }, { status: error.response?.status || 500 });
    }
    
    return NextResponse.json({ 
      error: 'Unexpected error occurred',
      details: error instanceof Error ? error.message : String(error)
    }, { status: 500 });
  }
} 