// app/api/sessions/[sessionId]/route.ts
import { NextRequest } from 'next/server';
import axios from 'axios';

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> }
) {
  try {
    const { sessionId } = await params; // Await the dynamic parameters

    const token = request.headers.get('authorization');
    if (!token) {
      return new Response(JSON.stringify({ error: 'No authorization token provided' }), {
        status: 401,
      });
    }

    if (!sessionId) {
      return new Response(JSON.stringify({ error: 'Session ID is required' }), {
        status: 400,
      });
    }

    // Use a fallback if API_BASE_URL is not set
    const apiBaseUrl = process.env.API_BASE_URL || 'http://127.0.0.1:5000';
    console.log(`Making delete request to: ${apiBaseUrl}/sessions/${sessionId}`);

    // Delete the session
    await axios.delete(`${apiBaseUrl}/sessions/${sessionId}`, {
      headers: {
        'Authorization': token
      }
    });

    // Remove the automatic creation of a new session
    // const newSessionResponse = await axios.post(`${apiBaseUrl}/sessions`, {}, {
    //   headers: {
    //     'Authorization': token
    //   }
    // });

    // Return only the success message
    return new Response(JSON.stringify({
      message: 'Session deleted successfully'
    }), {
      status: 200,
    });

  } catch (error: any) {
    console.error('Error in DELETE session:', error);
    return new Response(JSON.stringify({ 
      error: error.response?.data?.error || 'Failed to delete session'
    }), {
      status: error.response?.status || 500,
    });
  }
}
