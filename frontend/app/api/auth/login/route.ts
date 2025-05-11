import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

export async function POST(request: NextRequest) {
  try {
    const { email, password } = await request.json();
    
    if (!email || !password) {
      return NextResponse.json({ 
        error: 'Email and password are required',
        code: 'MISSING_CREDENTIALS'
      }, { status: 400 });
    }

    const baseUrl = process.env.BACKEND_URL || 'http://127.0.0.1:5000';
    const response = await axios.post(`${baseUrl}/auth/login`, { email, password });

    return NextResponse.json(response.data);
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Login error:', error.response?.data);
      return NextResponse.json({ 
        error: 'Login failed',
        details: error.response?.data || error.message
      }, { status: error.response?.status || 500 });
    }
    
    return NextResponse.json({ 
      error: 'Unexpected error occurred',
      details: error instanceof Error ? error.message : String(error)
    }, { status: 500 });
  }
} 