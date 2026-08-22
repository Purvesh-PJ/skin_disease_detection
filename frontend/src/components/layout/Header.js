import React, { useState } from 'react';
import styled from 'styled-components';
import { Link } from 'react-router-dom';
import { FiLogOut, FiLogIn, FiActivity, FiHome, FiSettings } from 'react-icons/fi';

import { ThemeToggle } from '../common/ui';
import Dropdown from '../common/ui/Dropdown';
import { authService } from '../../services';
import { ROUTES } from '../../constants';
import DefaultProfile from '../../assets/images/default_profile.jpg';
import ProfileSettingsModal from '../features/profile/ProfileSettingsModal';

const HeaderContainer = styled.header`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[6]}`};
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.default};
`;

const HeaderLeft = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[3]};
`;

const HeaderRight = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[3]};
`;

const Logo = styled(Link)`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2.5] || '10px'};
  text-decoration: none;
`;

const LogoIcon = styled.div`
  width: 34px;
  height: 34px;
  background: ${({ theme }) => theme.colors.primary[600]};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  box-shadow: 0 2px 8px rgba(22, 163, 74, 0.25);
`;

const LogoText = styled.span`
  font-family: ${({ theme }) => theme.fontFamily?.heading || 'inherit'};
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: ${({ theme }) => theme.colors.text.primary};
  
  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    display: none;
  }
`;

const UserStatusBadge = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.8rem;
  padding: 4px 10px;
  border-radius: ${({ theme }) => theme.borderRadius.pill};
  background: ${({ theme }) => theme.colors.background.tertiary};
  color: ${({ theme }) => theme.colors.text.secondary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  font-weight: 600;

  span.dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: ${({ $isAuth }) => ($isAuth ? '#16a34a' : '#f59e0b')};
  }

  @media (max-width: ${({ theme }) => theme.breakpoints.xs}) {
    display: none;
  }
`;

const ProfileTrigger = styled.button`
  background: none;
  border: none;
  padding: 0;
  cursor: pointer;
  display: flex;
  align-items: center;
  outline: none;
`;

const ProfileImage = styled.img`
  width: 36px;
  height: 36px;
  border-radius: 50%;
  object-fit: cover;
  border: 2px solid ${({ theme }) => theme.colors.border.default};
  transition: border-color ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary[400]};
  }
`;

const Header = ({ onUserUpdated }) => {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const user = authService.getUser();

  const handleLogout = () => {
    authService.logout();
  };

  const displayName = user?.settings?.full_name || user?.username || 'Guest Evaluator';
  const roleTitle = user?.settings?.role_title || (user ? 'Verified User' : 'Demo Mode');

  return (
    <>
      <HeaderContainer>
        <HeaderLeft>
          <Logo to={ROUTES.HOME}>
            <LogoIcon>
              <FiActivity size={18} />
            </LogoIcon>
            <LogoText>Skin Disease AI</LogoText>
          </Logo>
        </HeaderLeft>

        <HeaderRight>
          <UserStatusBadge $isAuth={!!user}>
            <span className="dot" />
            <span>{user ? `${displayName}` : 'Demo / Guest'}</span>
          </UserStatusBadge>
          
          <ThemeToggle />
          
          <Dropdown.Root>
            <Dropdown.Trigger asChild>
              <ProfileTrigger aria-label="User account menu">
                <ProfileImage src={DefaultProfile} alt="Profile" />
              </ProfileTrigger>
            </Dropdown.Trigger>

            <Dropdown.Content align="end" sideOffset={8}>
              <Dropdown.Item style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '2px' }}>
                <div style={{ fontWeight: 700, fontSize: '0.88rem' }}>{displayName}</div>
                <div style={{ fontSize: '0.75rem', color: '#888' }}>{roleTitle}</div>
              </Dropdown.Item>

              <Dropdown.Separator />

              {user && (
                <Dropdown.Item onClick={() => setIsSettingsOpen(true)}>
                  <FiSettings size={14} />
                  Profile & MongoDB Settings
                </Dropdown.Item>
              )}

              <Dropdown.Item asChild>
                <Link to={ROUTES.HOME} style={{ textDecoration: 'none', color: 'inherit', display: 'flex', alignItems: 'center', gap: '8px', width: '100%' }}>
                  <FiHome size={14} />
                  Project Home
                </Link>
              </Dropdown.Item>

              <Dropdown.Separator />

              {user ? (
                <Dropdown.Item className="danger" onClick={handleLogout}>
                  <FiLogOut size={14} />
                  Logout
                </Dropdown.Item>
              ) : (
                <Dropdown.Item asChild>
                  <Link to={ROUTES.LOGIN} style={{ textDecoration: 'none', color: 'inherit', display: 'flex', alignItems: 'center', gap: '8px', width: '100%' }}>
                    <FiLogIn size={14} />
                    Sign in / 1-Click Demo
                  </Link>
                </Dropdown.Item>
              )}
            </Dropdown.Content>
          </Dropdown.Root>
        </HeaderRight>
      </HeaderContainer>

      <ProfileSettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onProfileUpdated={onUserUpdated}
      />
    </>
  );
};

export default Header;

