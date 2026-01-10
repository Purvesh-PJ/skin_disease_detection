import { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { FiLogOut, FiUser, FiSettings } from 'react-icons/fi';
import { ThemeToggle } from '../common/ui';
import { authService } from '../../services';
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
  gap: ${({ theme }) => theme.spacing[4]};
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
`;

const LogoIcon = styled.div`
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, ${({ theme }) => theme.colors.primary[500]}, ${({ theme }) => theme.colors.primary[700]});
  border-radius: ${({ theme }) => theme.borderRadius.md};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 700;
  font-size: 0.9rem;
`;

const LogoText = styled.span`
  font-size: 1.1rem;
  font-weight: 600;
  color: ${({ theme }) => theme.colors.text.primary};
  
  @media (max-width: ${({ theme }) => theme.breakpoints.sm}) {
    display: none;
  }
`;

const ProfileContainer = styled.div`
  position: relative;
  cursor: pointer;
`;

const ProfileImage = styled.img`
  width: 36px;
  height: 36px;
  border-radius: 50%;
  object-fit: cover;
  border: 2px solid ${({ theme }) => theme.colors.border.light};
  transition: border-color ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => theme.colors.primary[400]};
  }
`;

const DropdownMenu = styled.div`
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  min-width: 180px;
  background-color: ${({ theme }) => theme.colors.background.primary};
  border: 1px solid ${({ theme }) => theme.colors.border.default};
  border-radius: ${({ theme }) => theme.borderRadius.lg};
  box-shadow: ${({ theme }) => theme.shadows.card};
  padding: ${({ theme }) => theme.spacing[2]};
  z-index: 100;
`;

const DropdownItem = styled.div`
  display: flex;
  align-items: center;
  gap: ${({ theme }) => theme.spacing[2]};
  padding: ${({ theme }) => `${theme.spacing[2]} ${theme.spacing[3]}`};
  border-radius: ${({ theme }) => theme.borderRadius.md};
  color: ${({ theme }) => theme.colors.text.primary};
  font-size: 0.875rem;
  cursor: pointer;
  transition: background-color ${({ theme }) => theme.transitions.fast};

  &:hover {
    background-color: ${({ theme }) => theme.colors.interactive.hover};
  }

  &.danger {
    color: ${({ theme }) => theme.colors.error[600]};
    
    &:hover {
      background-color: ${({ theme }) => theme.colors.status.error.bg};
    }
  }
`;

const DropdownDivider = styled.hr`
  border: none;
  border-top: 1px solid ${({ theme }) => theme.colors.border.light};
  margin: ${({ theme }) => theme.spacing[2]} 0;
`;

const Header = () => {
  const [isDropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);
  const user = authService.getUser();

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = () => {
    authService.logout();
  };

  return (
    <HeaderContainer>
      <HeaderLeft>
        <Logo>
          <LogoIcon>SP</LogoIcon>
          <LogoText>Skin Disease Predictor</LogoText>
        </Logo>
      </HeaderLeft>

      <HeaderRight>
        <ThemeToggle />
        
        <ProfileContainer ref={dropdownRef} onClick={() => setDropdownOpen(!isDropdownOpen)}>
          <ProfileImage src={DefaultProfile} alt="Profile" />

          {isDropdownOpen && (
            <DropdownMenu>
              <DropdownItem>
                <FiUser size={14} />
                {user?.username || 'Profile'}
              </DropdownItem>
              <DropdownItem>
                <FiSettings size={14} />
                Settings
              </DropdownItem>
              <DropdownDivider />
              <DropdownItem className="danger" onClick={handleLogout}>
                <FiLogOut size={14} />
                Logout
              </DropdownItem>
            </DropdownMenu>
          )}
        </ProfileContainer>
      </HeaderRight>
    </HeaderContainer>
  );
};

export default Header;
