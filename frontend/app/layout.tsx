// app/layout.tsx

import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { AuthProvider } from '@/lib/auth'
import Navigation from '@/components/Navigation'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Codex',
  description: 'Your AI-powered coding assistant',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AuthProvider>
          <div className="flex flex-col h-screen">
            <Navigation />
            <main className="flex-1 overflow-hidden">{children}</main>
          </div>
        </AuthProvider>
      </body>
    </html>
  )
}