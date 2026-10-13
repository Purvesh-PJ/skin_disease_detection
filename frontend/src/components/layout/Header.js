import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { FiLogOut, FiLogIn, FiActivity, FiHome, FiSettings } from 'react-icons/fi';
import ThemeToggle from '../ui/ThemeToggle';
import Dropdown from '../ui/Dropdown';
import { authService } from '../../services';
import { ROUTES } from '../../constants';
import DefaultProfile from '../../assets/images/default_profile.jpg';
import ProfileSettingsModal from '../features/profile/ProfileSettingsModal';

const Header = ({ onUserUpdated }) => {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const user = authService.getUser();

  const handleLogout = () => {
    authService.logout();
  };

  const getSanitizedName = () => {
    if (!user) return 'Demo';
    if (user.email === 'demo@skindisease.ai' || user.is_demo) return 'Demo';
    const raw = user.settings?.full_name || user.username || 'Demo';
    if (raw.includes('Evaluator') || raw.includes('Recruiter') || raw.includes('Demo User')) return 'Demo';
    return raw;
  };

  const getSanitizedRole = () => {
    if (!user) return 'Demo';
    if (user.email === 'demo@skindisease.ai' || user.is_demo) return 'Demo';
    const raw = user.settings?.role_title || 'Demo';
    if (raw.includes('Evaluator') || raw.includes('Recruiter')) return 'Demo';
    return raw;
  };

  const displayName = getSanitizedName();
  const roleTitle = getSanitizedRole();

  return (
    <>
      <header className="sticky top-0 left-0 right-0 z-50 backdrop-blur-md bg-white/90 dark:bg-[#121212]/90 flex items-center justify-between px-6 py-3 border-b border-neutral-200/80 dark:border-neutral-800/80">
        <div className="flex items-center gap-3">
          <Link to={ROUTES.HOME} className="flex items-center gap-2.5 no-underline">
            <div className="w-8.5 h-8.5 bg-emerald-600 rounded-lg flex items-center justify-center text-white shadow-sm">
              <FiActivity size={18} />
            </div>
            <span className="font-sans font-bold text-lg tracking-tight text-neutral-900 dark:text-neutral-100 hidden sm:inline">
              Skin Disease AI
            </span>
          </Link>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden xs:flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 border border-neutral-200 dark:border-neutral-700">
            <span className={`w-2 h-2 rounded-full ${user ? 'bg-emerald-600' : 'bg-amber-500'}`} />
            <span>{displayName}</span>
          </div>

          <ThemeToggle />

          <Dropdown.Root>
            <Dropdown.Trigger asChild>
              <button aria-label="User account menu" className="p-0 border-none bg-transparent cursor-pointer flex items-center outline-none">
                <img
                  src={DefaultProfile}
                  alt="Profile"
                  className="w-9 h-9 rounded-full object-cover border-2 border-neutral-200 dark:border-neutral-700 hover:border-emerald-500 transition-colors"
                />
              </button>
            </Dropdown.Trigger>

            <Dropdown.Content align="end" sideOffset={8}>
              <Dropdown.Item className="flex-col items-start gap-0.5">
                <div className="font-bold text-sm text-neutral-900 dark:text-neutral-100">{displayName}</div>
                <div className="text-xs text-neutral-500 dark:text-neutral-400">{roleTitle}</div>
              </Dropdown.Item>

              <Dropdown.Separator />

              {user && (
                <Dropdown.Item onClick={() => setIsSettingsOpen(true)}>
                  <FiSettings size={14} className="text-neutral-500" />
                  <span>Profile & Account Settings</span>
                </Dropdown.Item>
              )}

              <Dropdown.Item asChild>
                <Link to={ROUTES.HOME} className="no-underline text-inherit flex items-center gap-2 w-full">
                  <FiHome size={14} className="text-neutral-500" />
                  <span>Project Home</span>
                </Link>
              </Dropdown.Item>

              <Dropdown.Separator />

              {user ? (
                <Dropdown.Item className="danger text-red-600" onClick={handleLogout}>
                  <FiLogOut size={14} />
                  <span>Logout</span>
                </Dropdown.Item>
              ) : (
                <Dropdown.Item asChild>
                  <Link to={ROUTES.LOGIN} className="no-underline text-inherit flex items-center gap-2 w-full">
                    <FiLogIn size={14} />
                    <span>Sign in / 1-Click Demo</span>
                  </Link>
                </Dropdown.Item>
              )}
            </Dropdown.Content>
          </Dropdown.Root>
        </div>
      </header>

      <ProfileSettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onProfileUpdated={onUserUpdated}
      />
    </>
  );
};

export default Header;
