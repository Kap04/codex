// app/api/sessions/[sessionId]/messages/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function GET(
  request: NextRequest,
  { params }: { params: { sessionId: string } }
) {
  try {
    const { sessionId } = params;
    const authHeader = request.headers.get('authorization');
    if (!authHeader) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    
    // Extract token from Authorization header
    const token = authHeader.startsWith('Bearer ') 
      ? authHeader.substring(7) 
      : authHeader;

    // Forward request to backend
    const response = await fetch(`${process.env.BACKEND_URL}/sessions/${sessionId}/messages`, {
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
    console.error(`Error in sessions/${params.sessionId}/messages API:`, error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: { sessionId: string } }
) {
  try {
    const { sessionId } = params;
    const authHeader = request.headers.get('authorization');
    if (!authHeader) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }
    
    // Extract token from Authorization header
    const token = authHeader.startsWith('Bearer ') 
      ? authHeader.substring(7) 
      : authHeader;

    const body = await request.json();

    // Forward request to backend
    const response = await fetch(`${process.env.BACKEND_URL}/sessions/${sessionId}/messages`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    });

    const data = await response.json();
    
    if (!response.ok) {
      return NextResponse.json({ error: data.error || 'Something went wrong' }, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error(`Error in sessions/${params.sessionId}/messages API:`, error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}