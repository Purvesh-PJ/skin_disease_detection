import React from 'react';
import styled from 'styled-components';
import { Link } from 'react-router-dom';
import { FiLogOut, FiUser, FiSettings, FiActivity, FiHome } from 'react-icons/fi';
import { ThemeToggle } from '../common/ui';
import Dropdown from '../common/ui/Dropdown';
import { authService } from '../../services';
import { ROUTES } from '../../constants';
import DefaultProfile from '../../assets/images/default_profile.jpg';

const HeaderContainer = styled.header`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${({ theme }) => `${theme.spacing[3]} ${theme.spacing[6]}`};
  background-color: ${({ theme }) => theme.colors.background.primary};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border.light};
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
  background: ${({ theme }) => theme.gradients.brandIcon};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25);
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

const Header = () => {
  const user = authService.getUser();

  const handleLogout = () => {
    authService.logout();
  };

  return (
    <HeaderContainer>
      <HeaderLeft>
        <Logo to={ROUTES.HOME}>
          <LogoIcon>
            <FiActivity size={18} />
          </LogoIcon>
          <LogoText>Skin AI Diagnostics</LogoText>
        </Logo>
      </HeaderLeft>

      <HeaderRight>
        <ThemeToggle />
        
        <Dropdown.Root>
          <Dropdown.Trigger asChild>
            <ProfileTrigger aria-label="User account menu">
              <ProfileImage src={DefaultProfile} alt="Profile" />
            </ProfileTrigger>
          </Dropdown.Trigger>

          <Dropdown.Content align="end" sideOffset={8}>
            <Dropdown.Item>
              <FiUser size={14} />
              {user?.username || 'Profile'}
            </Dropdown.Item>
            <Dropdown.Item asChild>
              <Link to={ROUTES.HOME} style={{ textDecoration: 'none', color: 'inherit', display: 'flex', alignItems: 'center', gap: '8px', width: '100%' }}>
                <FiHome size={14} />
                Landing Page
              </Link>
            </Dropdown.Item>
            <Dropdown.Item>
              <FiSettings size={14} />
              Settings
            </Dropdown.Item>
            <Dropdown.Separator />
            <Dropdown.Item className="danger" onClick={handleLogout}>
              <FiLogOut size={14} />
              Logout
            </Dropdown.Item>
          </Dropdown.Content>
        </Dropdown.Root>
      </HeaderRight>
    </HeaderContainer>
  );
};

export default Header;
