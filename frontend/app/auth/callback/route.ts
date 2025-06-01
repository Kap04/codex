import { NextResponse } from 'next/server'
import { createClient } from '@/utils/supabase/server'; // Import the server-side client

export async function GET(request: Request) {
  console.log('Server-side OAuth callback hit (simplified redirect):', request.url);
  const requestUrl = new URL(request.url);
  const code = requestUrl.searchParams.get('code');
  // We no longer rely on the 'next' parameter from the initial redirect
  const redirectTo = `${requestUrl.origin}/chat/new`; // Hardcode the post-auth redirect

  if (code) {
    // Create a server-side Supabase client
    const supabase = createClient();
    // Exchange the authorization code for a session
    const { error } = await (await supabase).auth.exchangeCodeForSession(code);

    if (!error) {
      console.log('Server-side callback: Successfully exchanged code for session, redirecting to', redirectTo);
      // Redirect to the intended page (middleware should now see the cookie)
      return NextResponse.redirect(redirectTo);
    } else {
      console.error('Server-side callback: Error exchanging code:', error);
      // Redirect to login with error if code exchange failed
      return NextResponse.redirect(`${requestUrl.origin}/login?error=${encodeURIComponent(error.message)}`);
    }
  }

  // If no code is present, redirect to login as a fallback
  console.log('Server-side callback: No code received, redirecting to login.');
  return NextResponse.redirect(`${requestUrl.origin}/login`);
}