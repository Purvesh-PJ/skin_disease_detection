import React from 'react';
import { FiSun, FiMoon } from 'react-icons/fi';
import { useTheme } from '../../context/ThemeContext';
import Tooltip from '../common/ui/Tooltip';

export const ThemeToggle = () => {
  const { isDarkMode, toggleTheme } = useTheme();

  return (
    <Tooltip content={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}>
      <button
        onClick={toggleTheme}
        aria-label="Toggle theme"
        type="button"
        className="flex items-center justify-center w-10 h-10 rounded-full border border-neutral-200 dark:border-neutral-800 bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 hover:bg-white dark:hover:bg-neutral-700 hover:text-emerald-600 dark:hover:text-emerald-400 transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
      >
        {isDarkMode ? <FiSun size={18} /> : <FiMoon size={18} />}
      </button>
    </Tooltip>
  );
};

export default ThemeToggle;
