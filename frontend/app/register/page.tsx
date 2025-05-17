'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/lib/auth';
import Link from 'next/link';

export default function RegisterPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [localError, setLocalError] = useState('');
  const { register, error: authError } = useAuth();

  // Update local error when auth error changes
  useEffect(() => {
    if (authError) {
      setLocalError(authError);
    }
  }, [authError]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError('');

    if (password !== confirmPassword) {
      setLocalError('Passwords do not match');
      return;
    }
    
    try {
      await register(email, password);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'An error occurred during registration');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#1C1C1C] py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-[#F5E8D8]">
            Create your account
          </h2>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="rounded-md shadow-sm -space-y-px">
            <div>
              <label htmlFor="email-address" className="sr-only">
                Email address
              </label>
              <input
                id="email-address"
                name="email"
                type="email"
                autoComplete="email"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-[#3A3A3A] placeholder-[#F5E8D8]/40 text-[#F5E8D8] bg-[#2A2A2A] rounded-t-md focus:outline-none focus:ring-[#DAA520] focus:border-[#DAA520] focus:z-10 sm:text-sm"
                placeholder="Email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="password" className="sr-only">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="new-password"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-[#3A3A3A] placeholder-[#F5E8D8]/40 text-[#F5E8D8] bg-[#2A2A2A] focus:outline-none focus:ring-[#DAA520] focus:border-[#DAA520] focus:z-10 sm:text-sm"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="confirm-password" className="sr-only">
                Confirm Password
              </label>
              <input
                id="confirm-password"
                name="confirm-password"
                type="password"
                autoComplete="new-password"
                required
                className="appearance-none rounded-none relative block w-full px-3 py-2 border border-[#3A3A3A] placeholder-[#F5E8D8]/40 text-[#F5E8D8] bg-[#2A2A2A] rounded-b-md focus:outline-none focus:ring-[#DAA520] focus:border-[#DAA520] focus:z-10 sm:text-sm"
                placeholder="Confirm Password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
              />
            </div>
          </div>

          {localError && (
            <div className="text-[#FF6F61] text-sm text-center p-2 bg-red-900/20 border border-red-700/30 rounded">
              <p className="font-medium">Error:</p>
              <p>{localError}</p>
            </div>
          )}

          <div>
            <button
              type="submit"
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-[#1C1C1C] bg-[#FF6F61] hover:bg-[#FF6F61]/90 focus:outline-none focus:ring-2 focus:ring-[#FF6F61]/50 transition-colors"
            >
              Sign up
            </button>
          </div>
        </form>

        <div className="text-sm text-center">
          <Link href="/login" className="font-medium text-[#DAA520] hover:text-[#DAA520]/90 transition-colors">
            Already have an account? Sign in
          </Link>
        </div>
      </div>
    </div>
  );
} 