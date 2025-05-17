"use client"

import { motion } from "framer-motion";
import { CheckCircle, Globe } from "lucide-react";
import Link from "next/link";

export default function CodexLanding() {
  return (
    <div className="min-h-screen bg-[#1C1C1C] text-[#F5E8D8] flex flex-col items-center justify-center p-6">
      <motion.h1
        className="text-5xl font-bold text-center mb-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        Welcome to <span className="text-[#FF6F61]">Codex</span>
      </motion.h1>
      <p className="text-lg text-[#F5E8D8]/80 text-center max-w-2xl mb-8">
        Codex enables users to create a personalized AI-powered documentation system simply by providing a URL as input.
      </p>
      
      <div className="flex flex-row gap-4">
        <Link 
          href="/register" 
          className="bg-[#FF6F61] hover:bg-[#FF6F61]/90 text-[#1C1C1C] px-5 py-3 rounded-lg transition-colors"
        >
          Sign up
        </Link>

        <Link 
          href="/login" 
          className="bg-[#DAA520] hover:bg-[#DAA520]/90 text-[#1C1C1C] px-5 py-3 rounded-lg transition-colors"
        >
          Login
        </Link>
      </div>

      <div className="mt-12 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-4xl">
        {features.map((feature, index) => (
          <motion.div
            key={index}
            className="bg-[#2A2A2A] p-6 rounded-xl shadow-md flex items-start gap-4 border border-[#3A3A3A]"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: index * 0.2 }}
          >
            <feature.icon className="text-[#DAA520] w-8 h-8" />
            <div>
              <h3 className="text-xl font-semibold text-[#F5E8D8]">{feature.title}</h3>
              <p className="text-[#F5E8D8]/60 mt-2">{feature.description}</p>
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
