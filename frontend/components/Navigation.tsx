'use client';

import Link from 'next/link';
import { useAuth } from '@/lib/auth';

export default function Navigation() {
  const { isAuthenticated, logout } = useAuth();

  return (
    <nav className="bg-[#1C1C1C] border-b border-[#2A2A2A]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-14">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <Link href="/" className="text-xl font-bold text-[#F5E8D8]">
                Codex
              </Link>
            </div>
          </div>

          <div className="flex items-center">
            {isAuthenticated ? (
              <button
                onClick={logout}
                className="ml-4 px-4 py-1.5 text-sm font-medium rounded-md text-[#1C1C1C] bg-[#DAA520] hover:bg-[#DAA520]/90 transition-colors focus:outline-none focus:ring-2 focus:ring-[#DAA520]/50"
              >
                Sign out
              </button>
            ) : (
              <div className="space-x-4">
                <Link
                  href="/login"
                  className="text-[#F5E8D8]/80 hover:text-[#F5E8D8] px-3 py-2 rounded-md text-sm font-medium transition-colors"
                >
                  Sign in
                </Link>
                <Link
                  href="/register"
                  className="px-4 py-1.5 text-sm font-medium rounded-md text-[#1C1C1C] bg-[#FF6F61] hover:bg-[#FF6F61]/90 transition-colors focus:outline-none focus:ring-2 focus:ring-[#FF6F61]/50"
                >
                  Sign up
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
} 