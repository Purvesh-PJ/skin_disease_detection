import React from 'react';
import { Link } from 'react-router-dom';
import { ROUTES } from '../../constants';
import { FiActivity } from 'react-icons/fi';
import ThemeToggle from '../ui/ThemeToggle';

export const AuthLayout = ({ children }) => {
  return (
    <div className="min-h-screen w-full flex flex-col bg-white dark:bg-[#121212] text-neutral-900 dark:text-neutral-100 transition-colors">
      {/* Header */}
      <header className="flex items-center justify-between px-4 sm:px-8 py-3 border-b border-neutral-200/80 dark:border-neutral-800/80">
        <Link to={ROUTES.HOME} className="flex items-center gap-2.5 text-inherit no-underline">
          <div className="w-8 h-8 rounded-lg bg-emerald-600 flex items-center justify-center text-white shadow-sm">
            <FiActivity size={18} />
          </div>
          <span className="font-sans font-bold text-lg tracking-tight text-neutral-900 dark:text-neutral-100">
            Skin Disease AI
          </span>
        </Link>
        <div className="flex items-center gap-4">
          <Link
            to={ROUTES.HOME}
            className="text-xs sm:text-sm font-semibold text-neutral-600 dark:text-neutral-400 hover:text-emerald-600 dark:hover:text-emerald-400 no-underline transition-colors"
          >
            ← Home
          </Link>
          <ThemeToggle />
        </div>
      </header>

      {/* Main Split Layout */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-8 grid grid-cols-1 md:grid-cols-2 items-center gap-8 md:gap-12">
        {/* Left Side: Medical Graphic Illustration */}
        <div className="hidden md:flex flex-col items-center text-center gap-4">
          <div className="w-56 h-56 flex items-center justify-center">
            <svg viewBox="0 0 200 200" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full drop-shadow-sm">
              <circle cx="100" cy="100" r="85" fill="#16a34a" fillOpacity="0.08" stroke="#16a34a" strokeWidth="2" strokeDasharray="4 4" />
              <rect x="50" y="55" width="100" height="90" rx="16" fill="#16a34a" fillOpacity="0.15" stroke="#16a34a" strokeWidth="2" />
              <circle cx="100" cy="100" r="28" fill="#16a34a" fillOpacity="0.25" />
              <circle cx="100" cy="100" r="14" fill="#16a34a" />
              <path d="M70 100H86M114 100H130M100 70V86M100 114V130" stroke="#16a34a" strokeWidth="2.5" strokeLinecap="round" />
              <path d="M60 40L75 55M140 40L125 55M60 160L75 145M140 160L125 145" stroke="#16a34a" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </div>

          <h2 className="text-2xl font-bold tracking-tight text-neutral-900 dark:text-neutral-100 m-0">
            Skin Lesion AI Classifier
          </h2>
          <p className="text-sm text-neutral-600 dark:text-neutral-400 max-w-sm leading-relaxed m-0">
            Fast, non-invasive dermoscopic screening powered by ensemble neural networks.
          </p>
        </div>

        {/* Right Side: Form Card Container */}
        <div className="flex justify-center w-full">
          <div className="w-full max-w-sm flex flex-col gap-3">
            {children}
          </div>
        </div>
      </main>
    </div>
  );
};

export default AuthLayout;
