import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { FiUser, FiBriefcase, FiAward, FiCheck, FiX, FiSave, FiDatabase } from 'react-icons/fi';
import { Spinner, Alert } from '../../common/ui';
import { authService } from '../../../services';

const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.65);
  backdrop-filter: blur(6px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: ${({ theme }) => theme.spacing[4]};
`;

const ModalCard = styled.div`
  background: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.xl};
  width: 100%;
  max-width: 520px;
  box-shadow: ${({ theme }) => theme.shadows.xl};
  overflow: hidden;
  display: flex;
  flex-direction: column;
  animation: fadeIn 0.2s ease-out;

  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: scale(0.96);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }
`;

const ModalHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${({ theme }) => `${theme.spacing[4]} ${theme.spacing[5]}`};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.default};

  .title-group {
    display: flex;
    align-items: center;
    gap: ${({ theme }) => theme.spacing[2.5]};

    h3 {
      font-size: 1.15rem;
      font-weight: 700;
      margin: 0;
      color: ${({ theme }) => theme.colors.text.primary};
    }
  }
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  color: ${({ theme }) => theme.colors.text.secondary};
  cursor: pointer;
  padding: 6px;
  border-radius: ${({ theme }) => theme.borderRadius.md};
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all ${({ theme }) => theme.transitions.fast};

  &:hover {
    color: ${({ theme }) => theme.colors.text.primary};
    background: ${({ theme }) => theme.colors.background.tertiary};
  }
`;

const ModalBody = styled.form`
  padding: ${({ theme }) => theme.spacing[5]};
  display: flex;
  flex-direction: column;
  gap: ${({ theme }) => theme.spacing[4]};
`;

const FormGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;

  label {
    font-size: 0.85rem;
    font-weight: 600;
    color: ${({ theme }) => theme.colors.text.secondary};
    display: flex;
    align-items: center;
    gap: 6px;
  }
`;

const Input = styled.input`
  width: 100%;
  box-sizing: border-box;
  padding: 10px 14px;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  background: ${({ theme }) => theme.colors.background.secondary};
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: 0.9rem;
  outline: none;
  transition: border-color ${({ theme }) => theme.transitions.fast};

  &:focus {
    border-color: ${({ theme }) => theme.colors.primary[500]};
  }
`;

const MongoNotice = styled.div`
  background: ${({ theme }) => theme.colors.background.tertiary};
  border: 1px solid ${({ theme }) => theme.colors.border.light};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  padding: 10px 12px;
  font-size: 0.8rem;
  color: ${({ theme }) => theme.colors.text.secondary};
  display: flex;
  align-items: center;
  gap: 8px;
`;

const ModalFooter = styled.div`
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: ${({ theme }) => theme.spacing[3]};
  padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[5]}`};
  border-top: 1px solid ${({ theme }) => theme.colors.border.default};
  background: ${({ theme }) => theme.colors.background.secondary};
`;

const ActionButton = styled.button`
  padding: 8px 16px;
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  font-size: 0.88rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all ${({ theme }) => theme.transitions.fast};
  border: none;

  &.cancel {
    background: transparent;
    color: ${({ theme }) => theme.colors.text.secondary};
    &:hover {
      background: ${({ theme }) => theme.colors.background.tertiary};
      color: ${({ theme }) => theme.colors.text.primary};
    }
  }

  &.save {
    background: ${({ theme }) => theme.colors.primary[600]};
    color: #ffffff;
    &:hover:not(:disabled) {
      background: ${({ theme }) => theme.colors.primary[700]};
    }
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

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
        setRoleTitle(user.settings?.role_title || 'Clinical AI Evaluator');
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
      setSuccess('Profile settings successfully updated in MongoDB!');
      if (onProfileUpdated) {
        onProfileUpdated();
      }
      setTimeout(() => {
        onClose();
      }, 900);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to save settings to MongoDB');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ModalOverlay onClick={onClose}>
      <ModalCard onClick={(e) => e.stopPropagation()}>
        <ModalHeader>
          <div className="title-group">
            <FiUser color="#16a34a" size={18} />
            <h3>Evaluator Profile & Preferences</h3>
          </div>
          <CloseButton onClick={onClose}>
            <FiX size={18} />
          </CloseButton>
        </ModalHeader>

        <ModalBody onSubmit={handleSubmit}>
          {success && <Alert variant="success"><FiCheck size={16} /> {success}</Alert>}
          {error && <Alert variant="error">{error}</Alert>}

          <MongoNotice>
            <FiDatabase size={16} color="#16a34a" />
            <span>Settings are persisted real-time to the MongoDB Atlas cluster.</span>
          </MongoNotice>

          <FormGroup>
            <label>
              <FiUser size={14} /> Full Name / Display Name
            </label>
            <Input
              type="text"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              placeholder="e.g. Dr. Alex Morgan / Sarah (Recruiter)"
              required
            />
          </FormGroup>

          <FormGroup>
            <label>
              <FiBriefcase size={14} /> Role / Title
            </label>
            <Input
              type="text"
              value={roleTitle}
              onChange={(e) => setRoleTitle(e.target.value)}
              placeholder="e.g. Senior AI Technical Recruiter"
              required
            />
          </FormGroup>

          <FormGroup>
            <label>
              <FiAward size={14} /> Clinical Focus / Specialization
            </label>
            <Input
              type="text"
              value={specialization}
              onChange={(e) => setSpecialization(e.target.value)}
              placeholder="e.g. Melanoma Screening, Ham10000 Benchmarks"
              required
            />
          </FormGroup>

          <ModalFooter>
            <ActionButton type="button" className="cancel" onClick={onClose} disabled={loading}>
              Cancel
            </ActionButton>
            <ActionButton type="submit" className="save" disabled={loading}>
              {loading ? <Spinner size="sm" color="white" /> : (
                <>
                  <FiSave size={14} />
                  Save to MongoDB
                </>
              )}
            </ActionButton>
          </ModalFooter>
        </ModalBody>
      </ModalCard>
    </ModalOverlay>
  );
};

export default ProfileSettingsModal;
