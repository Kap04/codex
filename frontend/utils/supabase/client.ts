import { createBrowserClient } from '@supabase/ssr'

export const createClient = () =>
  createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get(name: string) {
          const kc = typeof window !== 'undefined' ? document.cookie : '';
          const match = kc.match('(^|; )' + name + '=([^;]+)');
          return match ? match[2] : null;
        },
        set(name: string, value: string, options: any) {
          // Set the cookie with a maxAge for persistence (e.g., 7 days)
          // This maxAge should ideally match the token expiration on the backend
          const maxAge = 60 * 60 * 24 * 7; // 7 days in seconds
          document.cookie = `${name}=${value}; path=${options.path}; Max-Age=${maxAge}`;
        },
        remove(name: string, options: any) {
          // Remove the cookie by setting maxAge to 0
          document.cookie = `${name}=; path=${options.path}; Max-Age=0`;
        },
      },
    }
  ) 