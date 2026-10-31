import React from 'react';
import { Link } from 'react-router-dom';
import { FiActivity, FiHome, FiCpu } from 'react-icons/fi';
import ThemeToggle from '../ui/ThemeToggle';
import { ROUTES } from '../../constants';

const Header = () => {
  return (
    <header className="sticky top-0 left-0 right-0 z-50 backdrop-blur-md bg-white/90 dark:bg-[#121212]/90 flex items-center justify-between px-6 py-3 border-b border-neutral-200/80 dark:border-neutral-800/80">
      <div className="flex items-center gap-3">
        <Link to={ROUTES.HOME} className="flex items-center gap-2.5 no-underline">
          <div className="w-8.5 h-8.5 bg-emerald-600 rounded-lg flex items-center justify-center text-white shadow-sm">
            <FiActivity size={18} />
          </div>
          <div className="flex flex-col">
            <span className="font-sans font-bold text-lg leading-tight tracking-tight text-neutral-900 dark:text-neutral-100">
              Skin Disease AI
            </span>
            <span className="text-[0.68rem] text-neutral-500 dark:text-neutral-400 font-medium hidden sm:inline">
              Ensemble Inference Studio
            </span>
          </div>
        </Link>
      </div>

      <div className="flex items-center gap-3">
        <div className="hidden md:flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20">
          <FiCpu size={13} />
          <span>HAM10000 Ensemble • TFLite Active</span>
        </div>

        <Link
          to={ROUTES.HOME}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg text-neutral-700 dark:text-neutral-300 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors no-underline border border-neutral-200 dark:border-neutral-800"
        >
          <FiHome size={14} />
          <span>Project Home</span>
        </Link>

        <ThemeToggle />
      </div>
    </header>
  );
};

export default Header;
