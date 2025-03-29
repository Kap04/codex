// app/api/ask/route.ts
import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { question } = body;
    
    if (!question) {
      return NextResponse.json({ error: 'Question is required' }, { status: 400 });
    }
    
    // Forward the request to your Python backend
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:5000';
    const response = await axios.post(`${backendUrl}/ask`, { question });
    
    return NextResponse.json(response.data);
  } catch (error: any) {
    console.error('Error getting answer:', error);
    return NextResponse.json({ 
      error: 'Failed to get answer',
      details: error.response?.data || error.message 
    }, { status: 500 });
  }
}
