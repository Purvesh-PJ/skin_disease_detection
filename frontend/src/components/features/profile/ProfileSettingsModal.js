import React, { useState, useEffect } from 'react';
import { FiUser, FiBriefcase, FiAward, FiCheck, FiX, FiSave, FiDatabase } from 'react-icons/fi';
import { Spinner, Alert, Button, Input } from '../../ui';
import { authService } from '../../../services';

export const ProfileSettingsModal = ({ isOpen, onClose, onProfileUpdated }) => {
  const [fullName, setFullName] = useState('');
  const [roleTitle, setRoleTitle] = useState('');
  const [specialization, setSpecialization] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (isOpen) {
      const user = authService.getUser();
      if (user) {
        setFullName(user.settings?.full_name || user.username || '');
        setRoleTitle(user.settings?.role_title || 'Demo User');
        setSpecialization(user.settings?.specialization || 'Dermoscopy Analysis');
      }
      setSuccess(null);
      setError(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const updatedPayload = {
        full_name: fullName.trim(),
        role_title: roleTitle.trim(),
        specialization: specialization.trim(),
      };
      await authService.updateProfile(updatedPayload);
      setSuccess('Profile settings successfully updated!');
      if (onProfileUpdated) {
        onProfileUpdated();
      }
      setTimeout(() => {
        onClose();
      }, 900);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to save settings');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 bg-black/65 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-200"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="bg-white dark:bg-[#181818] border border-neutral-200 dark:border-neutral-800 rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden flex flex-col"
      >
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-200 dark:border-neutral-800">
          <div className="flex items-center gap-2.5">
            <FiUser className="text-emerald-600 dark:text-emerald-400" size={18} />
            <h3 className="text-lg font-bold text-neutral-900 dark:text-neutral-100 m-0">
              Profile & Preferences
            </h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-200 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition-colors bg-transparent border-none cursor-pointer"
          >
            <FiX size={18} />
          </button>
        </div>

        {/* Modal Body */}
        <form onSubmit={handleSubmit} className="p-6 flex flex-col gap-4">
          {success && (
            <Alert variant="success">
              <FiCheck size={16} /> {success}
            </Alert>
          )}
          {error && <Alert variant="error">{error}</Alert>}

          <div className="bg-neutral-50 dark:bg-[#121212] border border-neutral-200/80 dark:border-neutral-800/80 rounded-xl p-3 text-xs text-neutral-600 dark:text-neutral-400 flex items-center gap-2">
            <FiDatabase size={16} className="text-emerald-600 dark:text-emerald-400 shrink-0" />
            <span>Preferences and diagnostic history are synced securely with your account.</span>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-neutral-700 dark:text-neutral-300 flex items-center gap-1.5">
              <FiUser size={14} /> Full Name / Display Name
            </label>
            <Input
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              placeholder="e.g. Demo User"
              required
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-neutral-700 dark:text-neutral-300 flex items-center gap-1.5">
              <FiBriefcase size={14} /> Role / Title
            </label>
            <Input
              type="text"
              value={roleTitle}
              onChange={(e) => setRoleTitle(e.target.value)}
              placeholder="e.g. AI Research / Doctor"
              required
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold text-neutral-700 dark:text-neutral-300 flex items-center gap-1.5">
              <FiAward size={14} /> Clinical Focus / Specialization
            </label>
            <Input
              type="text"
              value={specialization}
              onChange={(e) => setSpecialization(e.target.value)}
              placeholder="e.g. Melanoma Screening, Ham10000 Benchmarks"
              required
            />
          </div>

          {/* Modal Footer */}
          <div className="flex items-center justify-end gap-3 pt-4 border-t border-neutral-200 dark:border-neutral-800 mt-2">
            <Button
              type="button"
              variant="secondary"
              onClick={onClose}
              disabled={loading}
            >
              Cancel
            </Button>
            <Button type="submit" variant="brand" disabled={loading}>
              {loading ? (
                <Spinner size="sm" color="white" />
              ) : (
                <div className="flex items-center gap-1.5">
                  <FiSave size={14} />
                  <span>Save Changes</span>
                </div>
              )}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ProfileSettingsModal;
