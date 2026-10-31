import React from 'react';
import { Link } from 'react-router-dom';
import { Button } from '../ui';
import ThemeToggle from '../ui/ThemeToggle';
import { ROUTES } from '../../constants';
import { FiActivity } from 'react-icons/fi';

const LandingNavbar = () => {
  return (
    <header className="sticky top-0 left-0 right-0 z-50 backdrop-blur-xl bg-white/90 dark:bg-[#121212]/90 border-b border-neutral-200/80 dark:border-neutral-800/80 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
        <Link to={ROUTES.HOME} className="flex items-center gap-2.5 no-underline">
          <div className="w-9 h-9 bg-emerald-600 rounded-lg flex items-center justify-center text-white shadow-sm">
            <FiActivity size={20} />
          </div>
          <span className="font-sans text-lg font-bold tracking-tight text-neutral-900 dark:text-neutral-100">
            Skin Disease AI
          </span>
        </Link>

        <nav className="hidden lg:flex items-center gap-6">
          <a href="#overview" className="text-sm font-medium text-neutral-600 dark:text-neutral-400 hover:text-emerald-600 dark:hover:text-emerald-400 no-underline transition-colors">
            Overview
          </a>
          <a href="#how-it-works" className="text-sm font-medium text-neutral-600 dark:text-neutral-400 hover:text-emerald-600 dark:hover:text-emerald-400 no-underline transition-colors">
            How It Works
          </a>
          <a href="#architecture" className="text-sm font-medium text-neutral-600 dark:text-neutral-400 hover:text-emerald-600 dark:hover:text-emerald-400 no-underline transition-colors">
            Architecture
          </a>
          <a href="#dataset" className="text-sm font-medium text-neutral-600 dark:text-neutral-400 hover:text-emerald-600 dark:hover:text-emerald-400 no-underline transition-colors">
            HAM10000 Dataset
          </a>
          <a href="#conditions" className="text-sm font-medium text-neutral-600 dark:text-neutral-400 hover:text-emerald-600 dark:hover:text-emerald-400 no-underline transition-colors">
            7 Conditions
          </a>
          <a href="#transparency" className="text-sm font-medium text-neutral-600 dark:text-neutral-400 hover:text-emerald-600 dark:hover:text-emerald-400 no-underline transition-colors">
            Project Scope
          </a>
        </nav>

        <div className="flex items-center gap-3">
          <ThemeToggle />
          <Button asChild variant="brand" size="sm">
            <Link to={ROUTES.DASHBOARD} className="no-underline text-white font-medium">
              Launch Diagnostic Studio
            </Link>
          </Button>
        </div>
      </div>
    </header>
  );
};

export default LandingNavbar;
