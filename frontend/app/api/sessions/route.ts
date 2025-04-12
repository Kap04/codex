// app/api/sessions/route.ts
import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    if (!authHeader) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    
    // Extract token from Authorization header
    const token = authHeader.startsWith('Bearer ') 
      ? authHeader.substring(7) 
      : authHeader;

    // Forward request to backend
    const response = await fetch(`${process.env.BACKEND_URL}/sessions`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    const data = await response.json();
    
    if (!response.ok) {
      return NextResponse.json({ error: data.error || 'Something went wrong' }, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error('Error in sessions API:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    if (!authHeader) {
      console.error('Authorization header is missing');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const token = authHeader.startsWith('Bearer ')
      ? authHeader.split(' ')[1]
      : authHeader;

    console.log('Creating new session with token');
    
    // Parse the request body or use an empty object if no body is provided
    const body = request.body ? await request.json() : {};
    
    const response = await axios.post(`${process.env.BACKEND_URL}/sessions`, body, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });

    console.log('Session created successfully:', response.data);
    return NextResponse.json(response.data);
  } catch (error) {
    console.error('Error creating session:', error);
    if (axios.isAxiosError(error)) {
      console.error('Axios error details:', {
        status: error.response?.status,
        data: error.response?.data,
        message: error.message
      });
      return NextResponse.json({ 
        error: 'Failed to create session',
        details: error.response?.data || error.message,
        status: error.response?.status
      }, { status: error.response?.status || 500 });
    }
    return NextResponse.json({ 
      error: 'Failed to create session',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}