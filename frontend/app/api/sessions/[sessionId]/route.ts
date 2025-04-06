// app/api/sessions/[sessionId]/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function DELETE(
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
    const response = await fetch(`${process.env.BACKEND_URL}/sessions/${sessionId}`, {
      method: 'DELETE',
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
    console.error(`Error in sessions/${params.sessionId} API:`, error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}