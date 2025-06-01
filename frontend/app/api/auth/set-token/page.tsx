'use client';

import { useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

export default function SetTokenPage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const token = searchParams.get('token');
    const next = searchParams.get('next') || '/chat/new';
    if (token) {
      // Set cookie (expires in 7 days)
      document.cookie = `token=${token}; path=/; max-age=${60 * 60 * 24 * 7}`;
      // Set localStorage for your React context
      localStorage.setItem('token', token);
      // Redirect to next page
      router.replace(next);
    } else {
      router.replace('/login');
    }
  }, [router, searchParams]);

  return <div>Signing you in...</div>;
}