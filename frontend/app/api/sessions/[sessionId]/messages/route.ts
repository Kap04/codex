// app/api/sessions/[sessionId]/messages/route.ts
import { NextRequest } from 'next/server';
import axios from 'axios';

export async function GET(
  request: Request,
  { params }: { params: Promise<{ sessionId: string }> }
) {
  try {
    const { sessionId } = await params;
    
    if (!sessionId) {
      throw new Error('Session ID is required');
    }
    
    console.log(`Attempting to fetch messages for session ID: ${sessionId}`);

    const authHeader = request.headers.get('authorization');
    if (!authHeader) {
      console.error('Authorization header is missing.');
      return Response.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const token = authHeader.startsWith('Bearer ')
      ? authHeader.split(' ')[1]
      : authHeader;

    const backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:5000';
    console.log(`Making request to: ${backendUrl}/sessions/${sessionId}/messages`);
    
    const response = await axios.get(`${backendUrl}/sessions/${sessionId}/messages`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    console.log('Response received:', response.status);
    return Response.json(response.data);
  } catch (error) {
    console.error('Detailed error:', error);
    if (axios.isAxiosError(error)) {
      console.error('Axios error details:', {
        status: error.response?.status,
        data: error.response?.data,
        message: error.message
      });
      return Response.json({ 
        error: 'Failed to fetch messages',
        details: error.response?.data || error.message,
        status: error.response?.status
      }, { status: error.response?.status || 500 });
    }
    return Response.json({ 
      error: 'Failed to fetch messages',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

export async function POST(
  request: Request,
  { params }: { params: Promise<{ sessionId: string }> }
) {
  try {
    const { sessionId } = await params;
    
    if (!sessionId) {
      throw new Error('Session ID is required');
    }

    console.log(`Creating message for session ID: ${sessionId}`);

    const authHeader = request.headers.get('authorization');
    if (!authHeader) {
      return Response.json({ error: 'Unauthorized' }, { status: 401 });
    }
    
    const token = authHeader.startsWith('Bearer ') 
      ? authHeader.split(' ')[1]
      : authHeader;

    const body = await request.json();
    const backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:5000';
    console.log(`Making request to: ${backendUrl}/sessions/${sessionId}/messages`);

    const response = await axios.post(`${backendUrl}/sessions/${sessionId}/messages`, body, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });

    console.log('Response received:', response.status);
    return Response.json(response.data);
  } catch (error) {
    console.error('Detailed error:', error);
    if (axios.isAxiosError(error)) {
      console.error('Axios error details:', {
        status: error.response?.status,
        data: error.response?.data,
        message: error.message
      });
      return Response.json({ 
        error: 'Failed to create message',
        details: error.response?.data || error.message,
        status: error.response?.status
      }, { status: error.response?.status || 500 });
    }
    return Response.json({ 
      error: 'Failed to create message',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}