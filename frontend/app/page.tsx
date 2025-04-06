"use client"

import { useState } from "react";
import { motion } from "framer-motion";
import { CheckCircle, Globe } from "lucide-react";
import Link from "next/link";




import { useRouter } from 'next/navigation'
import { useEffect } from 'react'

export default function CodexLanding() {
  

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-6">

      <motion.h1
        className="text-5xl font-bold text-center mb-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        Welcome to <span className="text-blue-400">Codex</span>
      </motion.h1>
      <p className="text-lg text-gray-300 text-center max-w-2xl mb-8">
        Codex enables users to create a personalized AI-powered documentation system simply by providing a URL as input.
      </p>
      
      <div className="flex flex-row ">

      <Link href="/register" className="bg-blue-500 hover:bg-blue-600 text-white px-5 py-3 rounded-lg">
        sign up
      </Link>

      <Link href="/login" className="bg-blue-500 hover:bg-blue-600 text-white px-5 py-3 rounded-lg">
        login
      </Link>
      </div>
        
      

      <div className="mt-12 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-4xl">
        {features.map((feature, index) => (
          <motion.div
            key={index}
            className="bg-gray-800 p-6 rounded-xl shadow-md flex items-start gap-4"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: index * 0.2 }}
          >
            <feature.icon className="text-blue-400 w-8 h-8" />
            <div>
              <h3 className="text-xl font-semibold">{feature.title}</h3>
              <p className="text-gray-400 mt-2">{feature.description}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

const features = [
  {
    title: "AI-Powered Documentation",
    description: "Automatically generate structured and insightful documentation from any URL.",
    icon: CheckCircle,
  },
  {
    title: "Real-Time Updates",
    description: "Keep your documentation fresh with automatic updates whenever content changes.",
    icon: Globe,
  },
  {
    title: "Easy Integration",
    description: "Seamlessly integrate with your existing workflow for a smooth experience.",
    icon: CheckCircle,
  },
];
